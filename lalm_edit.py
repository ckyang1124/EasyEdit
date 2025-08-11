from easyeditor import DeSTA25AudioDataset, Qwen2AudioDataset
from easyeditor import LALMTrainer

# from easyeditor.dataset.LALM_edit_dataset import DeSTA25AudioDataset, Qwen2AudioDataset
# from easyeditor.trainer.LALMTrainer import LALMTrainer
from easyeditor import MENDLALMTrainingHparams, MENDLALMHparams
from easyeditor import EFKLALMTrainingHparams
# from easyeditor.models.mend.mend_lalm_hparams import MENDLALMHparams

def train_EFK_DeSTA25_debug():
    hparams = EFKLALMTrainingHparams.from_hparams("hparams/TRAINING/EFK/desta25-audio.yaml")
    
    train_ds = DeSTA25AudioDataset("/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/debug_Animal_transcriptions.json", config=hparams)
    test_ds = DeSTA25AudioDataset("/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/debug_Animal_transcriptions.json", config=hparams)
    
    trainer = LALMTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=test_ds
    )
    trainer.run()
    
def train_EFK_Qwen2Audio_debug():
    hparams = EFKLALMTrainingHparams.from_hparams("hparams/TRAINING/EFK/qwen2-audio.yaml")
    
    train_ds = Qwen2AudioDataset("/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/debug_Animal_transcriptions.json", config=hparams)
    test_ds = Qwen2AudioDataset("/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/debug_Animal_transcriptions.json", config=hparams)
    
    trainer = LALMTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=test_ds
    )
    trainer.run()
        
def train_MEND_DeSTA25_debug():
    hparams = MENDLALMTrainingHparams.from_hparams('hparams/TRAINING/MEND/desta25-audio.yaml')
    
    train_ds = DeSTA25AudioDataset("/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/debug_Animal_transcriptions.json", config=hparams)
    test_ds = DeSTA25AudioDataset("/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/debug_Animal_transcriptions.json", config=hparams)
    
    trainer = LALMTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=test_ds
    )
    trainer.run()
    
def train_MEND_Qwen2Audio_debug():
    hparams = MENDLALMTrainingHparams.from_hparams('hparams/TRAINING/MEND/qwen2-audio.yaml')
    
    train_ds = Qwen2AudioDataset("/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/debug_Animal_transcriptions.json", config=hparams)
    test_ds = Qwen2AudioDataset("/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/debug_Animal_transcriptions.json", config=hparams)
    
    trainer = LALMTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=test_ds
    )
    trainer.run()
    
if __name__ == "__main__":
    # train_MEND_DeSTA25_debug()
    # train_MEND_Qwen2Audio_debug()

    train_EFK_DeSTA25_debug() 
    # train_EFK_Qwen2Audio_debug()   
    