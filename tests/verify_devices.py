import jax

def check_devices():
    devices = jax.devices()
    backend = jax.default_backend()
    
    print("\n" + "="*40)
    print(f"JAX Hardware Verification")
    print("="*40)
    print(f"Backend: {backend.upper()}")
    print(f"Devices found: {len(devices)}")
    for i, dev in enumerate(devices):
        print(f"  - Device {i}: {dev}")
    print("="*40 + "\n")

if __name__ == "__main__":
    check_devices()