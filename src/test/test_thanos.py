import toml

from salsa.externals.thanosQuery import ThanosQuery
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent  # salsa/
config_path = BASE_DIR / "salsa" / "config" / "config.toml"
print(config_path)
config = toml.load(config_path)
thanos_query = ThanosQuery(config)

val = thanos_query.run_query("svc_delay")
print("svc_delay: " + str(val))

val = thanos_query.run_query("svc_request_rate")
print("svc_request_rate:  " + str(val))

val = thanos_query.run_query("svc_internal_delay")
print("svc_internal_delay:  " + str(val))
