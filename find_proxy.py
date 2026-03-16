"""One-shot script to find your Polymarket proxy wallet address and verify credentials."""
import os, sys, requests
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from eth_account import Account
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType
from web3 import Web3
from eth_abi import encode

pk  = os.getenv("POLY_PRIVATE_KEY")
ak  = os.getenv("POLY_API_KEY")
aks = os.getenv("POLY_API_SECRET")
ap  = os.getenv("POLY_API_PASSPHRASE")

eoa = Account.from_key(pk).address
print(f"EOA: {eoa}")

# ── 1. Verify CLOB auth works ──
print("\n── Verifying CLOB authentication ──")
client = ClobClient(
    host="https://clob.polymarket.com", key=pk, chain_id=137, signature_type=1,
    creds=ApiCreds(api_key=ak, api_secret=aks, api_passphrase=ap),
)
bal_result = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
bal = int(bal_result.get("balance", "0"))
print(f"CLOB balance (sig_type=1): ${bal / 1e6:.2f}")
if bal == 0:
    print("ERROR: $0 balance. Check .env credentials.")
    sys.exit(1)
print("Auth OK ✅")

# ── 2. Find proxy wallet via Gnosis Safe CREATE2 derivation ──
print("\n── Deriving proxy wallet address ──")
rpc = "https://polygon-bor-rpc.publicnode.com"
factory = "0xaacFeEa03eb1561C4e67d661e40682Bd20E3541b"

# Get proxyCreationCode from factory
pc_sel = Web3.keccak(text="proxyCreationCode()")[:4].hex()
r = requests.post(rpc, json={"jsonrpc":"2.0","method":"eth_call",
    "params":[{"to":factory,"data":"0x"+pc_sel},"latest"],"id":1},
    timeout=15, headers={"Content-Type":"application/json"})
raw = bytes.fromhex(r.json()["result"][2:])
offset = int.from_bytes(raw[:32], "big")
length = int.from_bytes(raw[offset:offset+32], "big")
creation_code = raw[offset+32:offset+32+length]
print(f"Factory creation code: {len(creation_code)} bytes")

singletons = [
    "0x3E5c63644E683549055b9Be8653de26E0B4CD36E",
    "0xd9Db270c1B5E3Bd161E8c8503c55cEABeE709552",
]
fallbacks = [
    "0xf48f2B2d2a534e402487b3ee7C18c33Aec0Fe5e4",
    "0x0000000000000000000000000000000000000000",
]

usdc_addr = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
found_proxy = None

for singleton in singletons:
    if found_proxy:
        break
    init_code = creation_code + encode(["address"], [singleton])
    init_code_hash = Web3.keccak(init_code)

    for fh in fallbacks:
        if found_proxy:
            break
        setup_sel = Web3.keccak(
            text="setup(address[],uint256,address,bytes,address,address,uint256,address)"
        )[:4]
        setup_params = encode(
            ["address[]","uint256","address","bytes","address","address","uint256","address"],
            [[eoa], 1, "0x"+"0"*40, b"", fh, "0x"+"0"*40, 0, "0x"+"0"*40],
        )
        initializer = setup_sel + setup_params

        for salt_nonce in range(3):
            salt_input = Web3.keccak(initializer) + salt_nonce.to_bytes(32, "big")
            salt = Web3.keccak(salt_input)
            create2_input = b"\xff" + bytes.fromhex(factory[2:]) + salt + init_code_hash
            candidate = Web3.to_checksum_address(Web3.keccak(create2_input)[-20:])

            # Check USDC balance on-chain
            bal_sel = Web3.keccak(text="balanceOf(address)")[:4].hex()
            bal_data = "0x" + bal_sel + encode(["address"], [candidate]).hex()
            r2 = requests.post(rpc, json={"jsonrpc":"2.0","method":"eth_call",
                "params":[{"to":usdc_addr,"data":bal_data},"latest"],"id":2},
                timeout=10, headers={"Content-Type":"application/json"})
            on_chain_bal = int(r2.json().get("result","0x0"), 16)

            if on_chain_bal > 0:
                found_proxy = candidate
                print(f"\n✅ PROXY WALLET FOUND: {candidate}")
                print(f"   On-chain USDC: ${on_chain_bal / 1e6:.4f}")
                break

if not found_proxy:
    print("Standard derivation didn't match. Trying MagicLink factory...")
    factory2 = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"
    gp_sel = Web3.keccak(text="getProxy(address)")[:4].hex()
    gp_data = "0x" + gp_sel + encode(["address"], [eoa]).hex()
    r3 = requests.post(rpc, json={"jsonrpc":"2.0","method":"eth_call",
        "params":[{"to":factory2,"data":gp_data},"latest"],"id":3},
        timeout=10, headers={"Content-Type":"application/json"})
    res3 = r3.json().get("result","0x")
    if res3 and res3 != "0x" and len(res3) >= 42:
        candidate2 = Web3.to_checksum_address("0x" + res3[-40:])
        if candidate2 != "0x" + "0"*40:
            found_proxy = candidate2
            print(f"✅ PROXY WALLET (MagicLink): {candidate2}")

if found_proxy:
    # Update .env with proxy address
    with open(os.path.join(os.path.dirname(__file__), ".env"), "r") as f:
        content = f.read()
    if "POLY_PROXY_ADDRESS" not in content:
        with open(os.path.join(os.path.dirname(__file__), ".env"), "a") as f:
            f.write(f"POLY_PROXY_ADDRESS={found_proxy}\n")
    else:
        lines = content.split("\n")
        new_lines = [f"POLY_PROXY_ADDRESS={found_proxy}" if l.startswith("POLY_PROXY_ADDRESS=") else l for l in lines]
        with open(os.path.join(os.path.dirname(__file__), ".env"), "w") as f:
            f.write("\n".join(new_lines))
    print(f"\n.env updated with POLY_PROXY_ADDRESS={found_proxy}")
else:
    print("\n❌ Could not find proxy wallet automatically.")
    print("You can find it on Polymarket website or Polygonscan.")
