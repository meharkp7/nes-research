from src.carrier_intelligence.carrier_scheduler import CarrierScheduler


profiles = [
    {
        "layer_id": 0,
        "module_name": "layer0",
        "adjusted_quality": 0.2,
        "num_params": 100,
    },
    {
        "layer_id": 1,
        "module_name": "layer1",
        "adjusted_quality": 0.8,
        "num_params": 100,
    },
    {
        "layer_id": 2,
        "module_name": "layer2",
        "adjusted_quality": 0.5,
        "num_params": 100,
    },
    {
        "layer_id": 3,
        "module_name": "layer3",
        "adjusted_quality": 0.1,
        "num_params": 100,
    },
]


scheduler = CarrierScheduler(gamma=2.5)

allocations = scheduler.allocate(
    profiles,
    total_payload_bits=100,
)

for allocation in allocations:
    print(
        f"Layer {allocation.layer_id}: "
        f"quality={allocation.adjusted_quality}, "
        f"capacity={allocation.capacity}, "
        f"bits={allocation.allocated_bits}"
    )

print()
print("Summary:")
print(scheduler.summary(allocations))