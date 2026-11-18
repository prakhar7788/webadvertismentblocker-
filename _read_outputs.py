import json, sys

nb = json.load(open(sys.argv[1], encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    outs = c.get("outputs", [])
    texts = []
    for o in outs:
        if o.get("output_type") == "stream":
            texts.append("".join(o.get("text", [])))
        elif o.get("output_type") in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "text/plain" in d:
                t = d["text/plain"]
                texts.append("".join(t) if isinstance(t, list) else t)
        elif o.get("output_type") == "error":
            texts.append("\n[ERROR] " + "\n".join(o.get("traceback", [])))
    block = "".join(texts).strip()
    if block:
        print(f"\n===== cell {i} =====")
        print(block)
