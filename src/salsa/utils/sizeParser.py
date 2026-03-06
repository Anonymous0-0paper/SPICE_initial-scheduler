def parse_memory_bytes(quantity):
    units = {'Ki': 1024, 'Mi': 1024 ** 2, 'Gi': 1024 ** 3, 'Ti': 1024 ** 4}
    for unit, multiplier in units.items():
        if str(quantity).endswith(unit):
            return int(quantity[:-len(unit)]) * multiplier
    if str(quantity).endswith('m'):
        return int(quantity[:-1]) / 1000
    return int(quantity)

def parse_cpu(quantity):
    if str(quantity).endswith('m'):
        return float(quantity[:-1]) / 1000
    elif str(quantity).endswith('n'):
        return float(quantity[:-1]) / 1_000_000_000
    elif str(quantity).endswith('u'):
        return float(quantity[:-1]) / 1_000_000
    return float(quantity)
