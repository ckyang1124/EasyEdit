from statistics import mean

from easyeditor import DeSTA25AudioDataset, Qwen2AudioDataset, AudioFlamingo3Dataset
from easyeditor import LALMTrainer

# from easyeditor.dataset.LALM_edit_dataset import DeSTA25AudioDataset, Qwen2AudioDataset
# from easyeditor.trainer.LALMTrainer import LALMTrainer
from easyeditor import MENDLALMTrainingHparams, MENDLALMHparams
from easyeditor import EFKLALMTrainingHparams, EFKLALMHyperParams
from easyeditor import FTLALMHyperParams
from easyeditor import IKELALMHyperParams
# from easyeditor.models.mend.mend_lalm_hparams import MENDLALMHparams
from easyeditor import LALMEditor
from argparse import ArgumentParser

debug_train_path = "/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/debug_Animal_transcriptions.json"
debug_val_path = "/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/debug_Animal_transcriptions.json"

train_path = "/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/ALL_train_transcriptions_no_label.json"
val_path = "/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/train/ALL_val_transcriptions_no_label.json"

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
    
def train_MEND_AudioFlamingo3():
    hparams = MENDLALMTrainingHparams.from_hparams('hparams/TRAINING/MEND/audio-flamingo-3.yaml')
    
    train_ds = AudioFlamingo3Dataset(train_path, config=hparams)
    test_ds = AudioFlamingo3Dataset(val_path, config=hparams)
    
    trainer = LALMTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=test_ds
    )
    trainer.run()
    
    
# ==== Testing Functions ====

def single_edit_MEND_DeSTA25():
    hparams = MENDLALMHparams.from_hparams('hparams/MEND/desta25-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for track in ["Animal", "Emotion", "Language", "Gender"]:
        test_ds = DeSTA25AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/{track}_transcriptions_no_label.json", config=hparams, testing=True)
    
        editor.single_edit_dataset(
            test_ds,
            output_path=f"{hparams.archive}_{track}_single_edit_no_label_all.jsonl",
        )
        
def sequential_edit_MEND_DeSTA25():
    hparams = MENDLALMHparams.from_hparams('hparams/MEND/desta25-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for seq_ind in range(10):
        test_ds = DeSTA25AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{seq_ind}.json", config=hparams, testing=True)
    
        editor.sequential_edit_dataset(
            test_ds,
            output_path=f"{hparams.archive}_sequential_edit_fixed_{seq_ind}.jsonl",
            generate_pre_edit=True,
        )
        
def single_edit_MEND_Qwen2Audio():
    hparams = MENDLALMHparams.from_hparams('hparams/MEND/qwen2-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for track in ["Animal", "Emotion", "Language", "Gender"]:
        test_ds = Qwen2AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/{track}_transcriptions_no_label.json", config=hparams, testing=True)
    
        editor.single_edit_dataset(
            test_ds,
            output_path=f"{hparams.archive}_{track}_single_edit_no_label_all.jsonl",
        )
        
def sequential_edit_MEND_Qwen2Audio():
    hparams = MENDLALMHparams.from_hparams('hparams/MEND/qwen2-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for seq_ind in range(10):
        test_ds = Qwen2AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{seq_ind}.json", config=hparams, testing=True)
    
        editor.sequential_edit_dataset(
            test_ds,
            output_path=f"{hparams.archive}_sequential_edit_fixed_{seq_ind}.jsonl",
            generate_pre_edit=True,
        )
        
def single_edit_EFK_DeSTA25():
    hparams = EFKLALMHyperParams.from_hparams('hparams/EFK/desta25-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for track in ["Animal", "Emotion", "Language", "Gender"]:
        test_ds = DeSTA25AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/{track}_transcriptions_no_label.json", config=hparams, testing=True)
    
        editor.single_edit_dataset(
            test_ds,
            output_path=f"{hparams.archive}_{track}_single_edit.jsonl",
            generate_pre_edit=False, # have already generated pre-edit results, so no need to generate again to save time and cost
        )
        
def sequential_edit_EFK_DeSTA25():
    hparams = EFKLALMHyperParams.from_hparams('hparams/EFK/desta25-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for seq_ind in range(10):
        test_ds = DeSTA25AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{seq_ind}.json", config=hparams, testing=True)
    
        editor.sequential_edit_dataset(
            test_ds,
            output_path=f"{hparams.archive}_sequential_edit_fixed_{seq_ind}.jsonl",
            generate_pre_edit=False,
        )
        
def single_edit_EFK_Qwen2Audio():
    hparams = EFKLALMHyperParams.from_hparams('hparams/EFK/qwen2-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for track in ["Animal", "Emotion", "Language", "Gender"]:
        test_ds = Qwen2AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/{track}_transcriptions_no_label.json", config=hparams, testing=True)
    
        editor.single_edit_dataset(
            test_ds,
            output_path=f"{hparams.archive}_{track}_single_edit.jsonl",
            generate_pre_edit=False, # have already generated pre-edit results, so no need to generate again to save time and cost
        )
        
def sequential_edit_EFK_Qwen2Audio():
    hparams = EFKLALMHyperParams.from_hparams('hparams/EFK/qwen2-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for seq_ind in range(10):
        test_ds = Qwen2AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{seq_ind}.json", config=hparams, testing=True)
    
        editor.sequential_edit_dataset(
            test_ds,
            output_path=f"{hparams.archive}_sequential_edit_fixed_{seq_ind}.jsonl",
            generate_pre_edit=False,
        )
        
def single_edit_FT_last_layer_DeSTA25():
    hparams = FTLALMHyperParams.from_hparams('hparams/FT/desta25-audio_last_layer.yaml')
    editor = LALMEditor.from_hparams(hparams)
    for track in ["Animal", "Emotion", "Language", "Gender"]:
        test_ds = DeSTA25AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/{track}_transcriptions_no_label.json", config=hparams, testing=True)
    
        editor.single_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/FT/last_layer/DeSTA25Audio/{track}_single_edit.jsonl",
            generate_pre_edit=False, # have already generated pre-edit results, so no need to generate again to save time and cost
        )
        
def sequential_edit_FT_last_layer_DeSTA25():
    hparams = FTLALMHyperParams.from_hparams('hparams/FT/desta25-audio_last_layer.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for seq_ind in range(10):
        test_ds = DeSTA25AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{seq_ind}.json", config=hparams, testing=True)
    
        editor.sequential_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/FT/last_layer/DeSTA25Audio/sequential_edit_fixed_{seq_ind}.jsonl",
            generate_pre_edit=False,
        )

def single_edit_FT_last_layer_Qwen2Audio():
    hparams = FTLALMHyperParams.from_hparams('hparams/FT/qwen2-audio_last_layer.yaml')
    editor = LALMEditor.from_hparams(hparams)
    for track in ["Animal", "Emotion", "Language", "Gender"]:
        test_ds = Qwen2AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/{track}_transcriptions_no_label.json", config=hparams, testing=True)
    
        editor.single_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/FT/last_layer/Qwen2Audio/{track}_single_edit.jsonl",
            generate_pre_edit=False, # have already generated pre-edit results, so no need to generate again to save time and cost
        )
        
def sequential_edit_FT_last_layer_Qwen2Audio():
    hparams = FTLALMHyperParams.from_hparams('hparams/FT/qwen2-audio_last_layer.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for seq_ind in range(10):
        test_ds = Qwen2AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{seq_ind}.json", config=hparams, testing=True)
    
        editor.sequential_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/FT/last_layer/Qwen2Audio/sequential_edit_fixed_{seq_ind}.jsonl",
            generate_pre_edit=False,
        )
       
def single_edit_FT_connector_DeSTA25():
    hparams = FTLALMHyperParams.from_hparams('hparams/FT/desta25-audio_connector.yaml')
    editor = LALMEditor.from_hparams(hparams)
    for track in ["Animal", "Emotion", "Language", "Gender"]:
        test_ds = DeSTA25AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/{track}_transcriptions_no_label.json", config=hparams, testing=True)
    
        editor.single_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/FT/connector/DeSTA25Audio/{track}_single_edit.jsonl",
            generate_pre_edit=False, # have already generated pre-edit results, so no need to generate again to save time and cost
        )
        
def sequential_edit_FT_connector_DeSTA25():
    hparams = FTLALMHyperParams.from_hparams('hparams/FT/desta25-audio_connector.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for seq_ind in range(10):
        test_ds = DeSTA25AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{seq_ind}.json", config=hparams, testing=True)
    
        editor.sequential_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/FT/connector/DeSTA25Audio/sequential_edit_fixed_{seq_ind}.jsonl",
            generate_pre_edit=False,
        )
        
def single_edit_FT_connector_Qwen2Audio():
    hparams = FTLALMHyperParams.from_hparams('hparams/FT/qwen2-audio_connector.yaml')
    editor = LALMEditor.from_hparams(hparams)
    for track in ["Animal", "Emotion", "Language", "Gender"]:
        test_ds = Qwen2AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/{track}_transcriptions_no_label.json", config=hparams, testing=True)
    
        editor.single_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/FT/connector/Qwen2Audio/{track}_single_edit.jsonl",
            generate_pre_edit=False, # have already generated pre-edit results, so no need to generate again to save time and cost
        )
        
def sequential_edit_FT_connector_Qwen2Audio():
    hparams = FTLALMHyperParams.from_hparams('hparams/FT/qwen2-audio_connector.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for seq_ind in range(10):
        test_ds = Qwen2AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{seq_ind}.json", config=hparams, testing=True)
    
        editor.sequential_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/FT/connector/Qwen2Audio/sequential_edit_fixed_{seq_ind}.jsonl",
            generate_pre_edit=False,
        )
        
def single_edit_IKE_DeSTA25():
    hparams = IKELALMHyperParams.from_hparams('hparams/IKE/desta25-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    for track in ["Animal", "Emotion", "Language", "Gender"]:
        test_ds = DeSTA25AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/{track}_transcriptions_no_label.json", config=hparams, testing=True)
    
        editor.single_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/IKE/DeSTA25Audio/{track}_single_edit.jsonl",
            generate_pre_edit=False, # have already generated pre-edit results, so no need to generate again to save time and cost
        )
        
def sequential_edit_IKE_DeSTA25():
    hparams = IKELALMHyperParams.from_hparams('hparams/IKE/desta25-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for seq_ind in range(10):
        test_ds = DeSTA25AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{seq_ind}.json", config=hparams, testing=True)
    
        editor.sequential_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/IKE/DeSTA25Audio/sequential_edit_fixed_{seq_ind}.jsonl",
            generate_pre_edit=False,
        )
        
def single_edit_IKE_Qwen2Audio():
    hparams = IKELALMHyperParams.from_hparams('hparams/IKE/qwen2-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    for track in ["Animal", "Emotion", "Language", "Gender"]:
        test_ds = Qwen2AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/{track}_transcriptions_no_label.json", config=hparams, testing=True)
    
        editor.single_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/IKE/Qwen2Audio/{track}_single_edit.jsonl",
            generate_pre_edit=False, # have already generated pre-edit results, so no need to generate again to save time and cost
        )
        
def sequential_edit_IKE_Qwen2Audio():
    hparams = IKELALMHyperParams.from_hparams('hparams/IKE/qwen2-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for seq_ind in range(10):
        test_ds = Qwen2AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{seq_ind}.json", config=hparams, testing=True)
    
        editor.sequential_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/IKE/Qwen2Audio/sequential_edit_fixed_{seq_ind}.jsonl",
            generate_pre_edit=False,
        )
        
def single_edit_IKE_wo_examples_DeSTA25():
    hparams = IKELALMHyperParams.from_hparams('hparams/IKE_wo_examples/desta25-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    for track in ["Animal", "Emotion", "Language", "Gender"]:
        test_ds = DeSTA25AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/{track}_transcriptions_no_label.json", config=hparams, testing=True)
    
        editor.single_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/IKE_wo_examples/DeSTA25Audio/{track}_single_edit.jsonl",
            generate_pre_edit=False, # have already generated pre-edit results, so no need to generate again to save time and cost
        )
        
def sequential_edit_IKE_wo_examples_DeSTA25():
    hparams = IKELALMHyperParams.from_hparams('hparams/IKE_wo_examples/desta25-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for seq_ind in range(10):
        test_ds = DeSTA25AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{seq_ind}.json", config=hparams, testing=True)
    
        editor.sequential_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/IKE_wo_examples/DeSTA25Audio/sequential_edit_fixed_{seq_ind}.jsonl",
            generate_pre_edit=False,
        )
        
def single_edit_IKE_wo_examples_Qwen2Audio():
    hparams = IKELALMHyperParams.from_hparams('hparams/IKE_wo_examples/qwen2-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    for track in ["Animal", "Emotion", "Language", "Gender"]:
        test_ds = Qwen2AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/{track}_transcriptions_no_label.json", config=hparams, testing=True)
    
        editor.single_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/IKE_wo_examples/Qwen2Audio/{track}_single_edit.jsonl",
            generate_pre_edit=False, # have already generated pre-edit results, so no need to generate again to save time and cost
        )
        
def sequential_edit_IKE_wo_examples_Qwen2Audio():
    hparams = IKELALMHyperParams.from_hparams('hparams/IKE_wo_examples/qwen2-audio.yaml')
    editor = LALMEditor.from_hparams(hparams)
    
    for seq_ind in range(10):
        test_ds = Qwen2AudioDataset(f"/work/b10902133/data/lalm-knowledge-editing/dataset/metadata/test/sequential_edits_fixed/seq_{seq_ind}.json", config=hparams, testing=True)
    
        editor.sequential_edit_dataset(
            test_ds,
            output_path=f"/work/b10902133/data/lalm-knowledge-editing/EasyEdit/results/IKE_wo_examples/Qwen2Audio/sequential_edit_fixed_{seq_ind}.jsonl",
            generate_pre_edit=False,
        )
    
if __name__ == "__main__":
    # train_MEND_DeSTA25()
    # train_MEND_Qwen2Audio()
    train_MEND_AudioFlamingo3()

    # train_EFK_DeSTA25() 
    # train_EFK_Qwen2Audio()   
    
    # Test!
    # single_edit_MEND_DeSTA25()
    # sequential_edit_MEND_DeSTA25()
    # single_edit_MEND_Qwen2Audio()
    # sequential_edit_MEND_Qwen2Audio()

    # single_edit_EFK_DeSTA25()
    # sequential_edit_EFK_DeSTA25()
    # single_edit_EFK_Qwen2Audio()
    # sequential_edit_EFK_Qwen2Audio()
    
    # single_edit_FT_last_layer_DeSTA25()
    # sequential_edit_FT_last_layer_DeSTA25()
    # single_edit_FT_last_layer_Qwen2Audio()
    # sequential_edit_FT_last_layer_Qwen2Audio()
    
    # single_edit_FT_connector_DeSTA25()
    # sequential_edit_FT_connector_DeSTA25()
    # single_edit_FT_connector_Qwen2Audio()
    # sequential_edit_FT_connector_Qwen2Audio()
    
    # single_edit_IKE_DeSTA25()
    # sequential_edit_IKE_DeSTA25()
    # single_edit_IKE_Qwen2Audio()
    # sequential_edit_IKE_Qwen2Audio()
    
    # single_edit_IKE_wo_examples_DeSTA25()
    # sequential_edit_IKE_wo_examples_DeSTA25()
    # single_edit_IKE_wo_examples_Qwen2Audio()
    # sequential_edit_IKE_wo_examples_Qwen2Audio()
    
    # parser = ArgumentParser()
    # parser.add_argument("--model", type=str, required=True, help="Model to use: Qwen, DeSTA")
    # parser.add_argument("--method", type=str, required=True, help="Editing method to use: MEND, EFK, FT_last_layer, FT_connector, IKE, IKE_wo_examples")
    # args = parser.parse_args()
    
    # if args.model == "Qwen" and args.method == "MEND":
    #     sequential_edit_MEND_Qwen2Audio()
    # elif args.model == "DeSTA" and args.method == "MEND":
    #     sequential_edit_MEND_DeSTA25()
    # elif args.model == "Qwen" and args.method == "EFK":
    #     sequential_edit_EFK_Qwen2Audio()
    # elif args.model == "DeSTA" and args.method == "EFK":
    #     sequential_edit_EFK_DeSTA25()
    # elif args.model == "Qwen" and args.method == "FT_last_layer":
    #     sequential_edit_FT_last_layer_Qwen2Audio()
    # elif args.model == "DeSTA" and args.method == "FT_last_layer":
    #     sequential_edit_FT_last_layer_DeSTA25()
    # elif args.model == "Qwen" and args.method == "FT_connector":
    #     sequential_edit_FT_connector_Qwen2Audio()
    # elif args.model == "DeSTA" and args.method == "FT_connector":
    #     sequential_edit_FT_connector_DeSTA25()
    # elif args.model == "Qwen" and args.method == "IKE":
    #     sequential_edit_IKE_Qwen2Audio()
    # elif args.model == "DeSTA" and args.method == "IKE":
    #     sequential_edit_IKE_DeSTA25()
    # elif args.model == "Qwen" and args.method == "IKE_wo_examples":
    #     sequential_edit_IKE_wo_examples_Qwen2Audio()
    # elif args.model == "DeSTA" and args.method == "IKE_wo_examples":
    #     sequential_edit_IKE_wo_examples_DeSTA25()
    # else:
    #     raise ValueError("Invalid model or method argument!")