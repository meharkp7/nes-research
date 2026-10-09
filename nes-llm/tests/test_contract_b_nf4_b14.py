import hashlib,json,struct,tempfile,unittest
from pathlib import Path
from scripts.contract_b_nf4_b14 import DEFAULT_TENSOR_KEY,TEST_KEY_HEX,PAYLOAD_BITS,payload_bytes,envelope,to_bits,positions,code_at,set_code,embed,extract,read_header,read_tensor

class ContractBB14Tests(unittest.TestCase):
 def setUp(self): self.key=bytes.fromhex(TEST_KEY_HEX)
 def test_payload_envelope(self):
  p=payload_bytes(); e=envelope(p); self.assertEqual(len(p)*8,PAYLOAD_BITS); self.assertEqual(len(e),len(p)+16); self.assertEqual(e[-8:],hashlib.sha256(p).digest()[:8])
 def test_positions(self):
  a=positions(self.key,20000,100); b=positions(self.key,20000,500); self.assertEqual(len(set(b)),500); self.assertEqual(a,b[:100])
 def test_nibble_order(self):
  b=bytearray([0xAB,0x34]); self.assertEqual([code_at(b,i) for i in range(4)],[10,11,3,4]); set_code(b,0,2); set_code(b,1,7); self.assertEqual(b,bytearray([0x27,0x34]))
 def test_round_trip_and_pair_ids(self):
  p=payload_bytes(); packed=bytes((i*37+11)&255 for i in range(8192)); stego,pos,changed,checks=embed(packed,to_bits(envelope(p)),self.key)
  got,detail=extract(stego,self.key); self.assertEqual(got,p); self.assertEqual(detail["carrier_count"],10128); self.assertEqual(len(pos),10128); self.assertTrue(all(checks.values())); self.assertLessEqual(changed,len(pos))
 def test_wrong_key(self):
  p=payload_bytes(); packed=bytes((i*13+9)&255 for i in range(8192)); stego,_,_,_=embed(packed,to_bits(envelope(p)),self.key)
  wrong=bytes.fromhex("ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100")
  with self.assertRaisesRegex(ValueError,"magic mismatch"): extract(stego,wrong)
 def test_corruption_rejected(self):
  p=payload_bytes(); packed=bytes((i*19+5)&255 for i in range(8192)); stego,pos,_,_=embed(packed,to_bits(envelope(p)),self.key); damaged=bytearray(stego)
  q=pos[64]; set_code(damaged,q,code_at(damaged,q)^1)
  with self.assertRaisesRegex(ValueError,"checksum mismatch"): extract(bytes(damaged),self.key)
 def test_safetensors_parser(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/"model.safetensors"; packed=bytes(range(256))*4; h={DEFAULT_TENSOR_KEY:{"dtype":"U8","shape":[len(packed),1],"data_offsets":[0,len(packed)]}}
   encoded=json.dumps(h,separators=(",",":")).encode()
   with path.open("wb") as f: f.write(struct.pack("<Q",len(encoded))); f.write(encoded); f.write(packed)
   header,start=read_header(path); self.assertGreater(start,8)
   _,_,record,lo,hi,actual=read_tensor(path,DEFAULT_TENSOR_KEY); self.assertEqual(record["shape"],[len(packed),1]); self.assertEqual((lo,hi),(0,len(packed))); self.assertEqual(actual,packed)
if __name__=="__main__": unittest.main()
