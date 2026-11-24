"""XSTest (Roettger et al., NAACL 2024): 250 safe + 200 unsafe prompts, base vs v1 vs v2 adapters.
Greedy generation with the model's chat template, max 160 new tokens, batch 16, left padding,
stop on config EOS + chat turn-end tokens. Raw outputs are saved; refusal is labelled on the box."""
import csv, json, os, time, torch
from huggingface_hub import hf_hub_download, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL="CohereLabs/tiny-aya-global"
p=hf_hub_download("Paul/XSTest","xstest_prompts.csv",repo_type="dataset")
rows=list(csv.DictReader(open(p,encoding="utf-8")))
print("prompts",len(rows),flush=True)
v2=snapshot_download("nightmoose/schemasmith-v2-adapters",allow_patterns=["v2-s*/*"])
V1S,V1D="nightmoose/schemasmith-tiny-aya-sft","nightmoose/schemasmith-tiny-aya-dpo"
systems=[("base",None,None),("v1/sft",V1S,None),("v1/dpo",V1D,V1S)]
for s in (1,2,3):
    sft=os.path.join(v2,f"v2-s{s}","sft-adapter"); dpo=os.path.join(v2,f"v2-s{s}","dpo-adapter")
    systems += [(f"v2-s{s}/sft",sft,None),(f"v2-s{s}/dpo",dpo,sft)]
tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
texts=[tok.apply_chat_template([{"role":"user","content":r["prompt"]}],tokenize=False,add_generation_prompt=True) for r in rows]
order=sorted(range(len(texts)),key=lambda i:len(texts[i]))
out={"benchmark":"Paul/XSTest xstest_prompts.csv (450 prompts)","generation":"greedy, chat template, max 160 new tokens, batch 16","rows":rows,"systems":{}}
for name,ad,mg in systems:
    t=time.time()
    m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda")
    if mg: m=PeftModel.from_pretrained(m,mg).merge_and_unload()
    if ad: m=PeftModel.from_pretrained(m,ad).merge_and_unload()
    m.eval()
    stops={m.generation_config.eos_token_id} if isinstance(m.generation_config.eos_token_id,int) else set(m.generation_config.eos_token_id or [])
    stops.add(tok.eos_token_id)
    for tkn in ("<|END_RESPONSE|>","<|END_OF_TURN_TOKEN|>"):
        i=tok.convert_tokens_to_ids(tkn)
        if isinstance(i,int) and i!=tok.unk_token_id: stops.add(i)
    res=[""]*len(texts)
    for s in range(0,len(order),16):
        idx=order[s:s+16]
        enc=tok([texts[i] for i in idx],return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g=m.generate(**enc,max_new_tokens=160,do_sample=False,eos_token_id=sorted(stops),pad_token_id=tok.pad_token_id)
        for j,i in enumerate(idx): res[i]=tok.decode(g[j,enc["input_ids"].shape[1]:],skip_special_tokens=True)
    out["systems"][name]={"adapter":ad,"merge_first":mg,"outputs":res,"seconds":round(time.time()-t,1)}
    print("SAFETY",name,out["systems"][name]["seconds"],repr(res[0][:80]),flush=True)
    del m; torch.cuda.empty_cache()
os.makedirs("/tmp/out",exist_ok=True); json.dump(out,open("/tmp/out/safety.json","w"),ensure_ascii=False)
print("SAFETY_DONE",flush=True)
