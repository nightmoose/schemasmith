"""Belebele 0-shot (lm-eval-harness prompt), log-likelihood over A-D, base vs v1 SFT vs v1 DPO."""
import json, os, time, torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL="CohereLabs/tiny-aya-global"; SFT="nightmoose/schemasmith-tiny-aya-sft"; DPO="nightmoose/schemasmith-tiny-aya-dpo"
LANGS=["eng_Latn","spa_Latn","fra_Latn","deu_Latn","por_Latn","hin_Deva","zho_Hans","jpn_Jpan"]
N=int(os.environ.get("N_PER_LANG","200"))
items=[]
for lg in LANGS:
    p=hf_hub_download("facebook/belebele",f"data/{lg}.jsonl",repo_type="dataset")
    rows=[json.loads(l) for l in open(p,encoding="utf-8")][:N]
    for i,r in enumerate(rows):
        prompt=(f"P: {r['flores_passage']}\nQ: {r['question'].strip()}\nA: {r['mc_answer1']}\nB: {r['mc_answer2']}\n"
                f"C: {r['mc_answer3']}\nD: {r['mc_answer4']}\nAnswer:")
        items.append({"lang":lg,"i":i,"link":r.get("link"),"qnum":r.get("question_number"),"prompt":prompt,"gold":int(r["correct_answer_num"])-1})
print("items",len(items),flush=True)
tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
cont=[tok(" "+c,add_special_tokens=False)["input_ids"] for c in "ABCD"]
print("choice token ids",cont,flush=True)
assert all(len(c)==1 for c in cont), "multi-token choices"
cid=[c[0] for c in cont]
def load(adapter=None, merge=None):
    m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda")
    if merge: m=PeftModel.from_pretrained(m,merge).merge_and_unload()
    if adapter: m=PeftModel.from_pretrained(m,adapter).merge_and_unload()
    return m.eval()
out={"benchmark":"facebook/belebele","revision_note":"first %d questions per language"%N,"langs":LANGS,"prompt":"lm-eval-harness belebele 0-shot (P/Q/A-D/Answer:), loglik of ' A'..' D', no chat template","systems":{}}
for name,ad,mg in (("base",None,None),("sft",SFT,None),("dpo",DPO,SFT)):
    t=time.time(); m=load(ad,mg); preds=[]
    order=sorted(range(len(items)),key=lambda k:len(items[k]["prompt"]))
    res=[None]*len(items)
    for s in range(0,len(order),4):
        idx=order[s:s+4]
        enc=tok([items[k]["prompt"] for k in idx],return_tensors="pt",padding=True).to("cuda")
        with torch.no_grad(): lg=m(**enc,logits_to_keep=1).logits[:,-1,:].float()
        lp=torch.log_softmax(lg,-1)[:,cid]
        for j,k in enumerate(idx): res[k]=int(lp[j].argmax())
    acc={lg_:0 for lg_ in LANGS}
    for it,p in zip(items,res): acc[it["lang"]]+= int(p==it["gold"])
    out["systems"][name]={"adapter":ad,"merge_first":mg,"pred":res,"correct_by_lang":acc,"n_by_lang":N,"accuracy":sum(int(p==it["gold"]) for it,p in zip(items,res))/len(items),"seconds":round(time.time()-t,1)}
    print("RET",name,out["systems"][name]["accuracy"],acc,flush=True)
    del m; torch.cuda.empty_cache()
out["gold"]=[it["gold"] for it in items]; out["ids"]=[[it["lang"],it["i"],it["link"],it["qnum"]] for it in items]
os.makedirs("/tmp/out",exist_ok=True); json.dump(out,open("/tmp/out/retention.json","w"),ensure_ascii=False)
print("RET_DONE",flush=True)
