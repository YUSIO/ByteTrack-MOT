"""Adapter-only checkpoint bound to an immutable detector checkpoint."""
import hashlib
from pathlib import Path
import torch


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def save_adapter(model,path,base_checkpoint,metadata):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.save(dict(adapter=model.adapter.state_dict(),architecture=model.architecture,
                    input_size=model.input_size,base_sha256=sha256(base_checkpoint),fusion_mode=model.adapter.fusion_mode,
                    metadata=metadata),path)


def load_adapter(model,path,base_checkpoint):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['base_sha256']!=sha256(base_checkpoint):
        raise ValueError('Frozen detector checkpoint hash mismatch')
    if state['architecture']!=model.architecture or tuple(state['input_size'])!=model.input_size:
        raise ValueError('Adapter architecture/geometry mismatch')
    mode=state.get('fusion_mode','full')
    if mode not in ('full','visual_only'):raise ValueError('Unknown checkpoint fusion mode')
    model.adapter.load_state_dict(state['adapter'],strict=True)
    model.adapter.fusion_mode=mode
    return state['metadata']
