from pypylon import pylon

def clean(s):
    s = str(s).strip() if s is not None else ""
    return None if s.upper() in ("", "N/A", "NA") else s

tl = pylon.TlFactory.GetInstance()
for d in tl.EnumerateDevices():
    fname  = d.GetFriendlyName()
    model  = d.GetModelName()
    serial = d.GetSerialNumber()

    ip   = clean(getattr(d, "GetIpAddress", lambda: None)())
    mac  = clean(getattr(d, "GetMacAddress", lambda: None)())
    mask = clean(getattr(d, "GetSubnetMask", lambda: None)())

    iface = "GigE" if ip else "USB"
    print(f"{fname} | {model} | SN={serial} | {iface}"
          f"{' IP='+ip if ip else ''} | MAC={mac or 'N/A'} | Mask={mask or 'N/A'}")