#!/usr/bin/env python3
"""Contract-B B1.4 10,000-bit NF4 artifact-only mechanism pilot.
Test-only key/payload; not production crypto. Run from nes-llm:
python scripts/contract_b_nf4_b14.py embed --original ../cache/contract_b_nf4_probe_retry --output-dir ../cache/contract_b_nf4_b14_10k
python scripts/contract_b_nf4_b14.py receive --stego-dir ../cache/contract_b_nf4_b14_10k --output-payload ../cache/b14.bin --expected-sha256 HASH
"""
from __future__ import annotations
import argparse, hashlib, hmac, json, os, platform, shutil, struct, sys
from pathlib import Path

MAGIC=b"NB14"; DOMAIN=b"NES-B1.4-carrier-v1\x00"; NONCE=b"NES-B1.4-test-nonce-v1"
PAYLOAD_BITS=10000; PAYLOAD_BYTES=PAYLOAD_BITS//8; CHECKSUM_BYTES=8
TEST_KEY_HEX="00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
DEFAULT_TENSOR_KEY="model.layers.0.self_attn.q_proj.weight"

def read_header(path):
    with Path(path).open("rb") as f:
        prefix=f.read(8)
        if len(prefix)!=8: raise ValueError(f"Invalid safetensors file: {path}")
        n=struct.unpack("<Q",prefix)[0]
        if n<=0 or n>Path(path).stat().st_size-8: raise ValueError("Invalid safetensors header length")
        return json.loads(f.read(n)),8+n

def find_shard(folder,tensor_key):
    folder=Path(folder); idx=folder/"model.safetensors.index.json"
    if idx.exists():
        mapping=json.loads(idx.read_text())["weight_map"]
        if tensor_key not in mapping: raise KeyError(f"{tensor_key} absent from index")
        path=folder/mapping[tensor_key]
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    path=folder/"model.safetensors"
    if not path.is_file(): raise FileNotFoundError(f"No model.safetensors/index in {folder}")
    return path

def read_tensor(path,tensor_key):
    header,data_start=read_header(path); rec=header.get(tensor_key)
    if not isinstance(rec,dict): raise KeyError(f"Tensor {tensor_key!r} missing")
    if rec.get("dtype")!="U8": raise TypeError(f"Expected packed U8, got {rec.get('dtype')!r}")
    start,end=rec["data_offsets"]
    if start<0 or end<=start: raise ValueError("Invalid tensor offsets")
    with Path(path).open("rb") as f: f.seek(data_start+start); packed=f.read(end-start)
    if len(packed)!=end-start: raise IOError("Short read of packed tensor")
    return data_start,rec,start,end,packed

def code_at(packed,i):
    x=packed[i//2]
    return (x>>4)&15 if i%2==0 else x&15

def set_code(packed,i,code):
    j=i//2; x=packed[j]
    packed[j]=((x&15)|((code&15)<<4)) if i%2==0 else ((x&240)|(code&15))

def positions(key,code_count,count,tensor_key=DEFAULT_TENSOR_KEY):
    if len(key)<16: raise ValueError("Test key must be at least 16 bytes")
    if code_count<=0 or count<0 or count>code_count: raise ValueError("Invalid carrier count")
    out=[]; seen=set(); counter=0
    while len(out)<count:
        block=hmac.new(key,DOMAIN+NONCE+tensor_key.encode()+counter.to_bytes(8,"big"),hashlib.sha256).digest(); counter+=1
        for off in range(0,len(block),4):
            p=int.from_bytes(block[off:off+4],"big")%code_count
            if p not in seen:
                seen.add(p); out.append(p)
                if len(out)==count: break
    return out

def payload_bytes():
    out=bytearray(); i=0
    while len(out)<PAYLOAD_BYTES:
        out.extend(hashlib.sha256(b"NES-B1.4-synthetic-payload-v1"+i.to_bytes(8,"big")).digest()); i+=1
    return bytes(out[:PAYLOAD_BYTES])

def envelope(payload):
    return MAGIC+len(payload).to_bytes(4,"big")+payload+hashlib.sha256(payload).digest()[:CHECKSUM_BYTES]

def to_bits(data): return [(b>>s)&1 for b in data for s in range(7,-1,-1)]
def from_bits(bits):
    if len(bits)%8: raise ValueError("Bits not byte aligned")
    out=bytearray()
    for i in range(0,len(bits),8):
        v=0
        for b in bits[i:i+8]: v=(v<<1)|int(b)
        out.append(v)
    return bytes(out)

def embed(packed,bits,key,tensor_key=DEFAULT_TENSOR_KEY):
    pos=positions(key,len(packed)*2,len(bits),tensor_key); out=bytearray(packed); changed=0
    for p,b in zip(pos,bits):
        old=code_at(packed,p); new=(old&14)|int(b); changed+=new!=old; set_code(out,p,new)
    chosen=set(pos)
    checks={"carrier_parity_matches":all((code_at(out,p)&1)==b for p,b in zip(pos,bits)),
      "pair_ids_preserved":all(code_at(packed,p)//2==code_at(out,p)//2 for p in pos),
      "only_selected_codes_changed":all(code_at(packed,i)==code_at(out,i) or i in chosen for i in range(len(packed)*2))}
    if not all(checks.values()): raise AssertionError(checks)
    return bytes(out),pos,changed,checks

def extract(packed,key,tensor_key=DEFAULT_TENSOR_KEY):
    count=len(packed)*2; hp=positions(key,count,64,tensor_key)
    head=from_bits([code_at(packed,p)&1 for p in hp])
    if head[:4]!=MAGIC: raise ValueError("Envelope magic mismatch (wrong key or incompatible artifact)")
    n=int.from_bytes(head[4:8],"big")
    if n<=0 or n>count//8: raise ValueError(f"Implausible payload length {n}")
    total=8+n+CHECKSUM_BYTES; pos=positions(key,count,total*8,tensor_key)
    data=from_bits([code_at(packed,p)&1 for p in pos])
    if data[:4]!=MAGIC or int.from_bytes(data[4:8],"big")!=n: raise ValueError("Envelope header mismatch")
    payload=data[8:8+n]
    if data[8+n:]!=hashlib.sha256(payload).digest()[:CHECKSUM_BYTES]: raise ValueError("Payload checksum mismatch")
    return payload,{"magic_valid":True,"length_valid":True,"checksum_valid":True,"carrier_count":len(pos)}

def sender(a):
    original=Path(a.original).resolve(); output=Path(a.output_dir).resolve()
    if output.exists(): raise FileExistsError(f"{output} exists; choose a fresh directory")
    if not original.is_dir(): raise FileNotFoundError(original)
    required=sum(p.stat().st_size for p in original.rglob("*") if p.is_file())+512*1024*1024
    if shutil.disk_usage(output.parent).free<required: raise OSError(f"Need about {required:,} free bytes")
    key=bytes.fromhex(a.test_key_hex); shard=find_shard(original,a.tensor_key)
    data_start,rec,start,end,packed=read_tensor(shard,a.tensor_key)
    payload=payload_bytes(); wrapped=envelope(payload); stego,pos,changed,checks=embed(packed,to_bits(wrapped),key,a.tensor_key)
    output.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(original,output)
    target=output/shard.relative_to(original)
    with target.open("r+b") as f: f.seek(data_start+start); f.write(stego); f.flush(); os.fsync(f.fileno())
    _,_,_,_,back=read_tensor(target,a.tensor_key); checks["serialized_readback_exact"]=back==stego
    report={"stage":"B1.4-sender","status":"PASS" if all(checks.values()) else "FAIL","original_checkpoint":str(original),
      "stego_checkpoint":str(output),"tensor_key":a.tensor_key,"shard":str(shard.relative_to(original)),
      "packed_tensor_shape":rec.get("shape"),"payload_bits":len(payload)*8,"payload_bytes":len(payload),
      "payload_sha256":hashlib.sha256(payload).hexdigest(),"envelope_bytes":len(wrapped),
      "envelope_overhead_bytes":len(wrapped)-len(payload),"carrier_count":len(pos),"changed_code_count":changed,"checks":checks,
      "test_key_id_sha256_prefix":hashlib.sha256(key).hexdigest()[:16],
      "interpretation":"Sender-side checks only; separate receiver and model reload are required."}
    rp=output.parent/(output.name+"_b14_sender_report.json"); rp.write_text(json.dumps(report,indent=2,sort_keys=True))
    print(f"[B1.4 sender] Status: {report['status']}"); print(f"[B1.4 sender] Payload SHA-256: {report['payload_sha256']}")
    print(f"[B1.4 sender] Carriers: {len(pos)}; changed codes: {changed}"); print(f"[B1.4 sender] Report: {rp}")
    return 0 if report["status"]=="PASS" else 2

def reload_check(folder,key,tensor_key,device,expected):
    import torch
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    module=tensor_key.removesuffix(".weight")
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(str(folder),quantization_config=q,device_map={"":device},trust_remote_code=True)
    modules=dict(model.named_modules())
    if module not in modules: raise KeyError(f"Missing reloaded module {module}")
    weight=modules[module].weight; qs=getattr(weight,"quant_state",None)
    if qs is None or getattr(qs,"quant_type",None)!="nf4": raise RuntimeError("Reloaded weight is not NF4")
    raw=bytes(weight.data.detach().cpu().contiguous().reshape(-1).tolist()); got,detail=extract(raw,key,tensor_key)
    if got!=expected: raise ValueError("Loaded model extraction differs from raw artifact extraction")
    return {"reloaded_model_nf4":True,"reloaded_model_payload_matches":True,"reloaded_model_carrier_count":detail["carrier_count"]}

def receiver(a):
    folder=Path(a.stego_dir).resolve()
    if not folder.is_dir(): raise FileNotFoundError(folder)
    key=bytes.fromhex(a.test_key_hex); shard=find_shard(folder,a.tensor_key); _,_,_,_,packed=read_tensor(shard,a.tensor_key)
    payload,detail=extract(packed,key,a.tensor_key); digest=hashlib.sha256(payload).hexdigest()
    matches=None if not a.expected_sha256 else digest.lower()==a.expected_sha256.lower()
    if matches is False: raise ValueError("Recovered payload SHA-256 differs from sender hash")
    output=Path(a.output_payload).resolve(); output.parent.mkdir(parents=True,exist_ok=True); output.write_bytes(payload)
    loaded={}
    if not a.skip_model_reload: print(f"[B1.4 receiver] Reloading stego model on {a.device}..."); loaded=reload_check(folder,key,a.tensor_key,a.device,payload)
    checks={"magic_valid":detail["magic_valid"],"length_valid":detail["length_valid"],"checksum_valid":detail["checksum_valid"],
      "payload_is_10000_bits":len(payload)*8==PAYLOAD_BITS,"expected_sha256_matches":matches is not False,
      "model_reload_check":a.skip_model_reload or loaded.get("reloaded_model_payload_matches",False),**loaded}
    req=("magic_valid","length_valid","checksum_valid","payload_is_10000_bits","expected_sha256_matches","model_reload_check")
    report={"stage":"B1.4-receiver","status":"PASS" if all(checks[k] is True for k in req) else "FAIL",
      "stego_checkpoint":str(folder),"tensor_key":a.tensor_key,"payload_bits":len(payload)*8,"payload_bytes":len(payload),
      "payload_sha256":digest,"expected_sha256_supplied":a.expected_sha256 or None,"expected_sha256_matches":matches,
      "recovered_payload_file":str(output),"receiver_inputs":["stego_checkpoint","test_key","fixed_protocol"],
      "original_checkpoint_accessed":False,"residual_or_delta_sidecar_accessed":False,"checks":checks,
      "interpretation":"Artifact-only mechanism pilot for this artifact/configuration; not a security, stealth, utility, robustness or novelty claim.",
      "python":sys.version,"platform":platform.platform()}
    rp=output.parent/(output.name+"_b14_receiver_report.json"); rp.write_text(json.dumps(report,indent=2,sort_keys=True))
    print(f"[B1.4 receiver] Status: {report['status']}"); print(f"[B1.4 receiver] Payload SHA-256: {digest}"); print(f"[B1.4 receiver] Report: {rp}")
    return 0 if report["status"]=="PASS" else 2

def main():
    p=argparse.ArgumentParser(description=__doc__); subs=p.add_subparsers(dest="mode",required=True)
    e=subs.add_parser("embed"); e.add_argument("--original",required=True); e.add_argument("--output-dir",required=True)
    e.add_argument("--tensor-key",default=DEFAULT_TENSOR_KEY); e.add_argument("--test-key-hex",default=TEST_KEY_HEX); e.set_defaults(func=sender)
    r=subs.add_parser("receive"); r.add_argument("--stego-dir",required=True); r.add_argument("--output-payload",required=True)
    r.add_argument("--expected-sha256",default=""); r.add_argument("--tensor-key",default=DEFAULT_TENSOR_KEY)
    r.add_argument("--test-key-hex",default=TEST_KEY_HEX); r.add_argument("--device",choices=("cpu","mps"),default="cpu")
    r.add_argument("--skip-model-reload",action="store_true"); r.set_defaults(func=receiver)
    a=p.parse_args(); return a.func(a)
if __name__=="__main__": raise SystemExit(main())
