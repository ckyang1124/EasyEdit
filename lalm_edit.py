from easyeditor import DeSTA25AudioDataset, Qwen2AudioDataset
from easyeditor import LALMTrainer

# from easyeditor.dataset.LALM_edit_dataset import DeSTA25AudioDataset, Qwen2AudioDataset
# from easyeditor.trainer.LALMTrainer import LALMTrainer
from easyeditor import MENDLALMTrainingHparams, MENDLALMHparams
from easyeditor import EFKLALMTrainingHparams
# from easyeditor.models.mend.mend_lalm_hparams import MENDLALMHparams

debug_train_path = "/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/debug_Animal_transcriptions.json"
debug_val_path = "/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/debug_Animal_transcriptions.json"

train_path = "/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/ALL_train_transcriptions.json"
val_path = "/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/ALL_val_transcriptions.json"

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
    
if __name__ == "__main__":
    train_MEND_DeSTA25()
    # train_MEND_Qwen2Audio()

    # train_EFK_DeSTA25() 
    # train_EFK_Qwen2Audio()   
    