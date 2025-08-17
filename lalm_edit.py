from statistics import mean

from easyeditor import DeSTA25AudioDataset, Qwen2AudioDataset
from easyeditor import LALMTrainer

# from easyeditor.dataset.LALM_edit_dataset import DeSTA25AudioDataset, Qwen2AudioDataset
# from easyeditor.trainer.LALMTrainer import LALMTrainer
from easyeditor import MENDLALMTrainingHparams, MENDLALMHparams
from easyeditor import EFKLALMTrainingHparams
# from easyeditor.models.mend.mend_lalm_hparams import MENDLALMHparams
from easyeditor import LALMEditor

debug_train_path = "/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/debug_Animal_transcriptions.json"
debug_val_path = "/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/debug_Animal_transcriptions.json"

train_path = "/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/ALL_train_transcriptions.json"
val_path = "/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/ALL_val_transcriptions.json"

def print_result(metrics):
    rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
    print(f'rewrite_acc: {rewrite_acc}')
    
    for key in metrics[0]['post'].keys():
        if key.startswith('generality') and key.endswith('acc'):
            generality_acc = mean([m['post'][key].item() for m in metrics])
            print(f'{key}: {generality_acc}')
    
    for key in metrics[0]['post'].keys():
        if key.startswith('locality') and key.endswith('acc'):
            locality_acc = mean([m['post'][key].item() for m in metrics])
            print(f'{key}: {locality_acc}')
    
    portability_acc = mean([m['post']['portability_acc'].item() for m in metrics])
    print(f'portability_acc: {portability_acc}')


# ==== Training Functions ===

def train_EFK_DeSTA25():
    hparams = EFKLALMTrainingHparams.from_hparams("hparams/TRAINING/EFK/desta25-audio.yaml")
    
    train_ds = DeSTA25AudioDataset(train_path, config=hparams)
    test_ds = DeSTA25AudioDataset(val_path, config=hparams)
    
    trainer = LALMTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=test_ds
    )
    trainer.run()
    
def train_EFK_Qwen2Audio():
    hparams = EFKLALMTrainingHparams.from_hparams("hparams/TRAINING/EFK/qwen2-audio.yaml")
    
    train_ds = Qwen2AudioDataset(train_path, config=hparams)
    test_ds = Qwen2AudioDataset(val_path, config=hparams)
    
    trainer = LALMTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=test_ds
    )
    trainer.run()
        
def train_MEND_DeSTA25():
    hparams = MENDLALMTrainingHparams.from_hparams('hparams/TRAINING/MEND/desta25-audio.yaml')
    
    train_ds = DeSTA25AudioDataset(train_path, config=hparams)
    test_ds = DeSTA25AudioDataset(val_path, config=hparams)
    
    trainer = LALMTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=test_ds
    )
    trainer.run()
    
def train_MEND_Qwen2Audio():
    hparams = MENDLALMTrainingHparams.from_hparams('hparams/TRAINING/MEND/qwen2-audio.yaml')
    
    train_ds = Qwen2AudioDataset(train_path, config=hparams)
    test_ds = Qwen2AudioDataset(val_path, config=hparams)
    
    trainer = LALMTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=test_ds
    )
    trainer.run()
    
# ==== Testing Functions ====

def edit_MEND_DeSTA25():
    hparams = MENDLALMHparams.from_hparams('hparams/MEND/desta25-audio.yaml')
    
    test_ds = DeSTA25AudioDataset("/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/Animal_transcriptions.json", config=hparams, testing=True)
    editor = LALMEditor.from_hparams(hparams)
    metrics, edted_model, _ = editor.edit(
        test_ds,
        keep_original_weight=True,
    )
    print("Metrics for MEND DeSTA25:")
    print_result(metrics)
    print("Raw metrics:")
    print(metrics)
    
if __name__ == "__main__":
    # train_MEND_DeSTA25()
    # train_MEND_Qwen2Audio()

    # train_EFK_DeSTA25() 
    # train_EFK_Qwen2Audio()   
    
    
    edit_MEND_DeSTA25()