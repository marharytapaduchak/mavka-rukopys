"""Shared, notebook-independent RUKOPYS stage 2. Heavy imports are lazy."""
import csv
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

TYPES = ('handwritten', 'printed', 'formula', 'table', 'annotation', 'image', 'graph')
MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'
TEXT_PROMPT = ('Transcribe all text in this cropped Ukrainian document region in reading order. '
               'Preserve original spelling, punctuation, historical spelling and incomplete words. '
               'Do not correct, translate, explain or complete the text. Return only the transcription.')
PROMPTS = {
    'handwritten': TEXT_PROMPT,
    'printed': TEXT_PROMPT,
    'annotation': TEXT_PROMPT,
    'formula': 'Transcribe every mathematical expression in this crop as LaTeX. Preserve symbols and structure. Return only the transcription, without Markdown fences or explanations.',
    'table': 'Transcribe this table exactly. Separate cells with | and rows with newlines. Preserve empty cells and original spelling. Return only the table, without explanations or Markdown separator rows.',
}
CLASS_PROMPT = ('Classify the main content of this document crop. Return exactly one label: '
                'handwritten, printed, formula, table, annotation, image, graph. '
                'Use annotation for marginal notes, signatures or correction marks; formula for mathematical expressions; '
                'table for tabular content; graph for plotted charts; image for non-text illustrations. No explanation.')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def cer_summary(rows):
    scored = [r for r in rows if r.get('reference', '')]
    if not scored: return {'n': 0, 'corpus_raw_cer': None, 'mean_region_raw_cer': None}
    return {'n': len(scored),
            'corpus_raw_cer': sum(edit_distance(r['reference'], r['prediction']) for r in scored)/sum(len(r['reference']) for r in scored),
            'mean_region_raw_cer': sum(edit_distance(r['reference'], r['prediction'])/len(r['reference']) for r in scored)/len(scored)}


def valid_bbox(bbox, width, height):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4: raise ValueError('bbox must have four coordinates')
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in bbox): raise ValueError('Invalid bbox values')
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height): raise ValueError(f'bbox outside image: {bbox}')
    return [float(x1), float(y1), float(x2), float(y2)]


def make_split(records, revision, seed=42, fraction=.2):
    ids = [r['file_name'] for r in records]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate page IDs')
    train, validation = [], []
    for source in sorted({r['source'] for r in records}):
        group = sorted([r for r in records if r['source']==source], key=lambda r: digest([seed, r['file_name']]))
        n = min(len(group)-1, max(1, math.ceil(len(group)*fraction))) if len(group)>1 else 0
        validation.extend(r['file_name'] for r in group[:n]); train.extend(r['file_name'] for r in group[n:])
    return {'dataset_revision': revision, 'seed': seed, 'train': sorted(train), 'validation': sorted(validation),
            'limitations': ['Page-disjoint; writer independence and dictation text independence are not guaranteed.']}


def prepare_data(root, seed=42, existing_split=None, revision=None):
    """Prepare gold crops with atomic per-page receipts and reuse of old PNGs."""
    from huggingface_hub import dataset_info, hf_hub_download
    from PIL import Image, ImageChops
    root = Path(root); data = root/'data'; data.mkdir(parents=True, exist_ok=True)
    repo = 'UkrainianCatholicUniversity/rukopys'
    state_path = data/'preparation_state.json'
    state = read_json(state_path) if state_path.exists() else {}
    previous_project = read_json(root/'project.json') if (root/'project.json').exists() else {}
    started = time.perf_counter()

    def download(filename, selected_revision):
        # Cache hits avoid an online HEAD request for every image.
        try:
            return hf_hub_download(repo, filename, repo_type='dataset', revision=selected_revision, local_files_only=True)
        except Exception:
            print(f'  Downloading: {filename}', flush=True)
        for attempt in range(1, 4):
            try:
                return hf_hub_download(repo, filename, repo_type='dataset', revision=selected_revision, etag_timeout=15)
            except Exception as exc:
                print(f'  Download attempt {attempt}/3 failed: {type(exc).__name__}: {exc}', flush=True)
                if attempt == 3: raise
        raise RuntimeError('Unreachable')

    revision = revision or state.get('dataset_revision') or previous_project.get('dataset_revision')
    print('Preparing metadata (cached files are reused).', flush=True)
    if not revision:
        # Recover the revision of the already cached main metadata from the old run.
        try:
            cached = hf_hub_download(repo, 'train/metadata.jsonl', repo_type='dataset', local_files_only=True)
            parts = Path(cached).parts
            revision = parts[parts.index('snapshots')+1]
            print(f'Recovered cached dataset revision: {revision}', flush=True)
        except Exception:
            revision = dataset_info(repo).sha
    metadata = download('train/metadata.jsonl', revision)
    records = [json.loads(line) for line in Path(metadata).read_text(encoding='utf-8').splitlines() if line.strip()]
    split_path = Path(existing_split) if existing_split else data/'split.json'
    split = read_json(split_path) if split_path.exists() else make_split(records, revision, seed)
    train, val = set(split['train']), set(split['validation'])
    all_ids = {r['file_name'] for r in records}
    if train & val or train | val != all_ids: raise ValueError('Split must partition all metadata pages exactly')
    if split.get('dataset_revision', revision) != revision: raise ValueError('Split revision mismatch')
    config = {'schema': 2, 'dataset_revision': revision, 'metadata_sha256': file_digest(metadata), 'split_sha256': digest(split), 'root': str(root.absolute())}
    if state and state.get('config') != config: raise ValueError('Preparation settings changed. Keep the same root, dataset revision and split; do not overwrite existing progress.')
    atomic_json(data/'split.json', split)
    atomic_json(state_path, {'dataset_revision':revision,'config':config,'complete':False,'processed_pages':0,'total_pages':len(records)})
    receipts = data/'preparation_pages'; receipts.mkdir(exist_ok=True)
    outputs = {'train': [], 'validation': []}; rejected = []; pages = []
    created = reused = resumed = 0
    print(f'Total pages: {len(records)}; annotated regions: {sum(len(p.get("regions",[])) for p in records)}', flush=True)
    print('Existing PNGs are checked against their source crop before reuse. Completed-page receipts allow later resumes without downloading that page.', flush=True)

    for page_number, page in enumerate(records, 1):
        partition = 'train' if page['file_name'] in train else 'validation'
        receipt = receipts/(digest(page['file_name'])+'.json')
        page_signature = digest({'config':config,'page':page,'partition':partition})
        cached_page = read_json(receipt) if receipt.exists() else None
        if cached_page and cached_page.get('signature') == page_signature and Path(cached_page['page']['image_path']).exists() and all(Path(r['crop_path']).exists() for r in cached_page['rows']):
            # Metadata comes from an atomically committed page receipt.
            outputs[partition].extend(cached_page['rows']); rejected.extend(cached_page['rejected']); pages.append(cached_page['page']); resumed += 1
        else:
            print(f'[{page_number}/{len(records)}] {page["file_name"]} | {partition} | regions={len(page.get("regions",[]))}', flush=True)
            image_path = download('train/'+page['file_name'], revision)
            page_rows = []; page_rejected = []
            with Image.open(image_path) as im:
                image = im.convert('RGB'); width, height = image.size
                page_info = {'image':page['file_name'],'image_path':str(image_path),'split':partition,'source':page['source'],'width':width,'height':height}
                for index, region in enumerate(page.get('regions', [])):
                    try:
                        kind = region['type']
                        if kind not in TYPES: raise ValueError('Unknown type')
                        bbox = valid_bbox(region['bbox'], width, height)
                        text = region.get('text') or ''
                        if not isinstance(text,str): raise ValueError('Non-string text')
                    except (ValueError, KeyError) as exc:
                        page_rejected.append({'page':page['file_name'],'region':index,'reason':str(exc)})
                        continue
                    uid = digest([page['file_name'], index])[:24]
                    crop_path = data/'crops'/partition/(uid+'.png'); crop_path.parent.mkdir(parents=True,exist_ok=True)
                    expected = image.crop(tuple(bbox))
                    matches = False
                    if crop_path.exists():
                        try:
                            with Image.open(crop_path) as old:
                                old.load()
                                matches = old.mode=='RGB' and old.size==expected.size and ImageChops.difference(old,expected).getbbox() is None
                        except (OSError, ValueError):
                            matches = False
                        # Never overwrite an old file whose content cannot be confirmed.
                        if not matches:
                            raise ValueError(f'Existing crop differs from source or is damaged: {crop_path}. It has been preserved. Move this specific file aside, then retry.')
                    if matches: reused += 1
                    else:
                        temporary = crop_path.with_suffix('.png.tmp')
                        expected.save(temporary,format='PNG',compress_level=1)
                        temporary.replace(crop_path); created += 1
                    scorable = kind not in ('image','graph') and region.get('language')=='uk' and region.get('legibility')=='legible' and bool(text.strip())
                    page_rows.append({'id':uid,'page_id':page['file_name'],'region_index':index,'source':page['source'],'bbox':bbox,'type':kind,'reference':text,'scorable':scorable,'crop_path':str(crop_path),'crop_sha256':file_digest(crop_path)})
                    expected.close()
                image.close()
            atomic_json(receipt,{'signature':page_signature,'page':page_info,'rows':page_rows,'rejected':page_rejected})
            outputs[partition].extend(page_rows); rejected.extend(page_rejected); pages.append(page_info)
        region_count = sum(len(rows) for rows in outputs.values())
        atomic_json(state_path,{'dataset_revision':revision,'config':config,'complete':False,'processed_pages':page_number,'total_pages':len(records),'regions':region_count,'last_page':page['file_name']})
        if page_number % 10 == 0 or page_number == len(records):
            elapsed = (time.perf_counter()-started)/60
            print(f'Progress: {page_number}/{len(records)} pages | {region_count} valid crops | reused PNG={reused}, new PNG={created}, resumed pages={resumed} | {elapsed:.1f} min',flush=True)

    for name, rows in outputs.items(): atomic_json(data/(name+'.json'), rows)
    atomic_json(data/'pages.json', pages); atomic_json(data/'rejected.json', rejected)
    project = {**previous_project,'dataset_id':repo,'dataset_revision':revision,'model_id':MODEL_ID,'seed':split.get('seed',seed),'train_manifest_sha256':digest(outputs['train']),'validation_manifest_sha256':digest(outputs['validation'])}
    atomic_json(root/'project.json', project)
    atomic_json(state_path,{'dataset_revision':revision,'config':config,'complete':True,'processed_pages':len(records),'total_pages':len(records),'regions':sum(len(rows) for rows in outputs.values())})
    result = {name:len(rows) for name, rows in outputs.items()} | {'rejected':len(rejected),'reused_png':reused,'new_png':created,'resumed_pages':resumed}
    print('Preparation completed; final manifests saved.',flush=True)
    return result


def load_qwen(root, adapter=None, quantized=True, max_pixels=512*512):
    import torch
    import importlib.metadata
    from huggingface_hub import model_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    root = Path(root); project = read_json(root/'project.json')
    if not torch.cuda.is_available(): raise RuntimeError('Enable a CUDA GPU runtime')
    model_id = project['model_id']
    # Resolve once per project, then reuse the exact revision for all notebooks.
    if 'model_revision' not in project:
        project['model_revision'] = model_info(model_id).sha; atomic_json(root/'project.json', project)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {'revision': project['model_revision'], 'dtype': dtype, 'device_map': {'':0}, 'attn_implementation':'sdpa'}
    if quantized: kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id, revision=project['model_revision'], min_pixels=32*32*32, max_pixels=max_pixels)
    if adapter:
        cfg = read_json(Path(adapter)/'run_config.json')
        if cfg['model_revision'] != project['model_revision']: raise ValueError('Adapter model revision mismatch')
        if cfg['protocol']['max_pixels'] != max_pixels: raise ValueError('Use the adapter training pixel budget or explicitly implement a resolution ablation')
        processor = AutoProcessor.from_pretrained(adapter)
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    protocol = {'model_id': model_id, 'model_revision': project['model_revision'], 'quantized': quantized, 'dtype': str(dtype), 'max_pixels': max_pixels,
                'adapter_sha256': file_digest(Path(adapter)/'adapter_model.safetensors') if adapter else None,
                'packages': {x: importlib.metadata.version(x) for x in ('torch','transformers','peft','bitsandbytes','pillow')}}
    return model, processor, protocol


def conversation(crop, prompt, answer=None):
    messages = [{'role':'user','content':[{'type':'image','image':crop},{'type':'text','text':prompt}]}]
    if answer is not None: messages.append({'role':'assistant','content':[{'type':'text','text':answer}]})
    return messages


def generate(model, processor, crop, prompt, max_tokens=512):
    import torch
    batch = processor.apply_chat_template(conversation(crop,prompt), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
    batch.pop('token_type_ids', None); batch = batch.to(model.device)
    torch.cuda.synchronize(); start = time.perf_counter()
    with torch.inference_mode(): output = model.generate(**batch, max_new_tokens=max_tokens, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    ids = output[0, batch['input_ids'].shape[1]:]
    prediction = processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return {'prediction':prediction, 'seconds':time.perf_counter()-start, 'generated_tokens':ids.numel(), 'hit_token_limit':ids.numel()>=max_tokens}


def parse_type(text):
    label = text.strip().lower()
    return label if label in TYPES else None


def evaluate(root, model, processor, protocol, output, prompts=None, max_tokens=512):
    from PIL import Image
    rows = read_json(Path(root)/'data/validation.json'); prompts = prompts or PROMPTS
    config = {'manifest':digest(rows), 'protocol':protocol, 'prompts':prompts, 'class_prompt':CLASS_PROMPT, 'max_tokens':max_tokens}
    saved = read_json(output) if Path(output).exists() else {'config':config,'results':[]}
    if saved['config'] != config: raise ValueError('Existing evaluation has a different configuration')
    results = saved['results']
    if [r['id'] for r in results] != [r['id'] for r in rows[:len(results)]]: raise ValueError('Resume prefix differs')
    model.eval()
    for row in rows[len(results):]:
        if file_digest(row['crop_path']) != row['crop_sha256']: raise ValueError('Crop bytes changed')
        with Image.open(row['crop_path']) as im: crop=im.convert('RGB')
        classification = generate(model,processor,crop,CLASS_PROMPT,32)
        predicted_type = parse_type(classification['prediction'])
        # Oracle-type branch isolates transcription; predicted-type branch measures routing errors.
        oracle = generate(model,processor,crop,prompts[row['type']],max_tokens) if row['scorable'] else {'prediction':'','seconds':0,'hit_token_limit':False}
        if predicted_type in ('image','graph'): routed={'prediction':'','seconds':0,'hit_token_limit':False}
        elif predicted_type in prompts:
            routed = oracle if predicted_type==row['type'] and row['scorable'] else generate(model,processor,crop,prompts[predicted_type],max_tokens)
        else: routed={'prediction':'','seconds':0,'hit_token_limit':False}
        results.append({**row, 'prediction':oracle['prediction'], 'oracle':oracle, 'routed':routed, 'classification':classification, 'predicted_type':predicted_type})
        atomic_json(output, {'config':config,'complete':len(results)==len(rows),'results':results})
        print(f"{len(results)}/{len(rows)} | {row['source']} | {row['type']} -> {predicted_type}")
    summary = {'oracle':cer_summary([r for r in results if r['scorable']]),
               'routed':cer_summary([{**r,'prediction':r['routed']['prediction']} for r in results if r['scorable']]),
               'classification_accuracy':sum(r['type']==r['predicted_type'] for r in results)/len(results) if results else None,
               'invalid_labels':sum(r['predicted_type'] is None for r in results),
               'by_type':{t:cer_summary([r for r in results if r['type']==t and r['scorable']]) for t in TYPES},
               'by_source':{s:cer_summary([r for r in results if r['source']==s and r['scorable']]) for s in sorted({r['source'] for r in results})}}
    atomic_json(output, {'config':config,'complete':True,'results':results,'summary':summary})
    return summary


def training_examples(rows):
    examples=[]
    for row in rows:
        examples.append({**row,'task':'classification','prompt':CLASS_PROMPT,'answer':row['type']})
        if row['scorable']: examples.append({**row,'task':'transcription','prompt':PROMPTS[row['type']],'answer':row['reference']})
    return examples


def supervised_batch(processor, example, device):
    import torch
    from PIL import Image
    with Image.open(example['crop_path']) as im: crop=im.convert('RGB')
    prefix = processor.apply_chat_template(conversation(crop,example['prompt']), tokenize=True, add_generation_prompt=True,return_dict=True,return_tensors='pt')
    full = processor.apply_chat_template(conversation(crop,example['prompt'],example['answer']),tokenize=True,add_generation_prompt=False,return_dict=True,return_tensors='pt')
    n=prefix['input_ids'].shape[1]
    if not torch.equal(prefix['input_ids'],full['input_ids'][:,:n]): raise ValueError('Chat-template prefix mismatch')
    labels=full['input_ids'].clone(); labels[:,:n]=-100; labels[full['attention_mask']==0]=-100
    if not (labels!=-100).any(): raise ValueError('No supervised answer tokens')
    full['labels']=labels; full.pop('token_type_ids',None)
    return full.to(device)


def train(root, model, processor, protocol, run_name='gold_multitask', epochs=1, lr=1e-4, rank=8, accumulation=4, resume=None, save_every=50, max_regions=None):
    import torch
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import set_seed
    if epochs<1 or accumulation<1 or save_every<1: raise ValueError('Invalid training settings')
    root=Path(root); rows=read_json(root/'data/train.json'); val=read_json(root/'data/validation.json')
    if {r['page_id'] for r in rows}&{r['page_id'] for r in val}: raise ValueError('Train/validation page overlap')
    if max_regions: rows=sorted(rows,key=lambda r:digest([42,r['id']]))[:max_regions]
    for row in rows:
        if file_digest(row['crop_path']) != row['crop_sha256']: raise ValueError('Training crop bytes changed')
    examples=training_examples(rows)
    if not examples: raise ValueError('No training examples')
    seed=read_json(root/'project.json')['seed']; set_seed(seed)
    config={'protocol':protocol,'model_revision':protocol['model_revision'],'examples_sha256':digest(examples),'seed':seed,'epochs':epochs,'lr':lr,'rank':rank,'alpha':rank*2,'dropout':.05,'accumulation':accumulation,'prompts':PROMPTS,'class_prompt':CLASS_PROMPT,'task_weighting':'one classification example per region and one transcription example per scorable region'}
    run=root/'runs'/run_name
    if (run/'config.json').exists() and not resume: raise ValueError('Run exists: explicitly resume or choose a new run_name')
    run.mkdir(parents=True,exist_ok=True)
    if protocol['adapter_sha256']: raise ValueError('Training must start from the base model')
    if protocol['quantized']: model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    else:
        model.requires_grad_(False); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); model.enable_input_require_grads()
    targets=[n for n,_ in model.named_modules() if 'language_model' in n and n.rsplit('.',1)[-1] in ('q_proj','k_proj','v_proj','o_proj')]
    if not targets: raise ValueError('No language attention modules found')
    model=get_peft_model(model,LoraConfig(r=rank,lora_alpha=rank*2,lora_dropout=.05,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=0)
    fp16=protocol['dtype']=='torch.float16'; dtype=torch.float16 if fp16 else torch.bfloat16
    scaler=torch.amp.GradScaler('cuda',enabled=fp16,init_scale=1024)
    orders=[]
    for epoch in range(epochs):
        order=list(range(len(examples))); random.Random(seed+epoch).shuffle(order); orders.append(order)
    epoch=position=updates=0; history=[]
    if resume:
        checkpoint=Path(resume)
        if not (checkpoint/'COMPLETE').exists(): raise ValueError('Incomplete checkpoint')
        if read_json(checkpoint/'run_config.json')!=config: raise ValueError('Resume configuration mismatch')
        set_peft_model_state_dict(model,load_file(str(checkpoint/'adapter_model.safetensors')))
        state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        epoch,position,updates=state['epoch'],state['position'],state['updates']; history=state['history']
        random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    atomic_json(run/'config.json',config)
    def save():
        dest=run/f'checkpoint_{updates:06d}'; temp=run/f'.checkpoint_{updates:06d}.tmp'
        if dest.exists(): return dest
        temp.mkdir(exist_ok=True); model.save_pretrained(temp); processor.save_pretrained(temp); atomic_json(temp/'run_config.json',config)
        torch.save({'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'position':position,'updates':updates,'history':history,'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temp/'training_state.pt')
        (temp/'COMPLETE').write_text('OK'); temp.rename(dest); atomic_json(run/'latest.json',{'checkpoint':str(dest)})
        return dest
    model.train(); model.config.use_cache=False
    while epoch<epochs:
        order=orders[epoch]
        while position<len(order):
            indices=order[position:position+accumulation]
            for attempt in range(4):
                optimizer.zero_grad(set_to_none=True); losses=[]
                for index in indices:
                    batch=supervised_batch(processor,examples[index],model.device)
                    with torch.autocast('cuda',dtype=dtype): out=model(**batch,use_cache=False); loss=out.loss
                    if not torch.isfinite(loss): raise RuntimeError('Non-finite forward loss; progress has not advanced')
                    scaler.scale(loss/len(indices)).backward(); losses.append(loss.item()); del batch,out,loss
                scaler.unscale_(optimizer)
                finite=all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
                if not finite:
                    if not fp16: raise RuntimeError('Non-finite BF16 gradients')
                    scaler.step(optimizer); scaler.update(); continue
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); break
            else: raise RuntimeError('Repeated gradient overflow; resume from latest checkpoint')
            optimizer.zero_grad(set_to_none=True); position+=len(indices); updates+=1
            history.append({'epoch':epoch+1,'position':position,'update':updates,'loss':sum(losses)/len(losses)})
            if updates%10==0: print(history[-1])
            if updates%save_every==0: save()
        epoch+=1; position=0; save()
    model.eval(); return model,str(save())


def detect_pages(pages, weights, output, confidence=.25, image_size=1024):
    from doclayout_yolo import YOLOv10
    config={'weights_sha256':file_digest(weights),'pages_sha256':digest(pages),'confidence':confidence,'image_size':image_size}
    saved=read_json(output) if Path(output).exists() else {'config':config,'pages':[]}
    if saved['config']!=config: raise ValueError('Detection configuration mismatch')
    done=saved['pages']
    if [p['image'] for p in done]!=[p['image'] for p in pages[:len(done)]]: raise ValueError('Detection resume order mismatch')
    model=YOLOv10(str(weights))
    for page in pages[len(done):]:
        result=model.predict(page['image_path'],conf=confidence,imgsz=image_size,device='cuda:0',verbose=False)[0]
        h,w=result.orig_shape
        if (w,h)!=(page['width'],page['height']): raise ValueError('Unexpected original image dimensions')
        regions=[]
        for box,score,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
            box=[max(0,min(w,box[0])),max(0,min(h,box[1])),max(0,min(w,box[2])),max(0,min(h,box[3]))]
            valid_bbox(box,w,h)
            label=result.names[int(cls)]
            regions.append({'bbox':box,'confidence':score,'type':label if label in TYPES else None})
        done.append({**page,'regions':regions})
        atomic_json(output,{'config':config,'complete':len(done)==len(pages),'pages':done})
    return done


def transcribe_pages(pages, model, processor, protocol, output, max_tokens=1024):
    from PIL import Image
    if len({p['image'] for p in pages})!=len(pages): raise ValueError('Duplicate filenames')
    config={'detections_sha256':digest(pages),'image_sha256':{p['image']:file_digest(p['image_path']) for p in pages},'protocol':protocol,'prompts':PROMPTS,'class_prompt':CLASS_PROMPT,'max_tokens':max_tokens}
    saved=read_json(output) if Path(output).exists() else {'config':config,'pages':[]}
    if saved['config']!=config: raise ValueError('Inference resume configuration mismatch')
    done=saved['pages']
    if [p['image'] for p in done]!=[p['image'] for p in pages[:len(done)]]: raise ValueError('Inference resume order mismatch')
    model.eval()
    for page in pages[len(done):]:
        with Image.open(page['image_path']) as im: image=im.convert('RGB')
        regions=[]; diagnostics=[]
        for index,region in enumerate(page['regions']):
            bbox=valid_bbox(region['bbox'],*image.size); crop=image.crop(tuple(bbox))
            # Always classify with Qwen so behavior matches the validation protocol.
            cls=generate(model,processor,crop,CLASS_PROMPT,32); kind=parse_type(cls['prediction'])
            if kind is None: raise ValueError(f"Invalid classification on {page['image']} region {index}: {cls['prediction']!r}; no guessed label exported")
            pred=generate(model,processor,crop,PROMPTS[kind],max_tokens) if kind in PROMPTS else {'prediction':'','hit_token_limit':False,'seconds':0}
            regions.append({'bbox':bbox,'type':kind,'text':pred['prediction']})
            diagnostics.append({'region_index':index,'classification':cls,'transcription':pred})
        done.append({'image':page['image'],'regions':regions,'diagnostics':diagnostics})
        atomic_json(output,{'config':config,'complete':len(done)==len(pages),'pages':done})
        print(f"{len(done)}/{len(pages)} | {page['image']} | {len(regions)} regions")
    return done


def write_submission(pages, expected_images, path, allow_truncated=False):
    names=[p['image'] for p in pages]
    if len(set(names))!=len(names) or set(names)!=set(expected_images): raise ValueError('Expected exactly one record for every image')
    for page in pages:
        if not allow_truncated and any(d['transcription']['hit_token_limit'] for d in page.get('diagnostics',[])): raise ValueError('Truncated generation: increase budget and use a new output file')
        for r in page['regions']:
            if r['type'] not in TYPES or not isinstance(r['text'],str): raise ValueError('Invalid region')
            if r['type'] in ('image','graph') and r['text']: raise ValueError('Non-text region must have empty text')
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    by_name={p['image']:p for p in pages}
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image','regions']);writer.writeheader()
        for name in expected_images: writer.writerow({'image':name,'regions':json.dumps(by_name[name]['regions'],ensure_ascii=False)})
    return str(path)
