from transformers import AutoModelForSeq2SeqLM, AutoModelForMultipleChoice
import pytorch_lightning as pl
import torch.nn as nn

MAX_LEN = 512
NUM_LABELS = 4


class KDAModel(pl.LightningModule):

    def __init__(self, model):
        super(KDAModel, self).__init__()
        self.save_hyperparameters()
        self.model_name = model
        if 't5' in self.model_name:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.t5_max_length = MAX_LEN
            # Set up loss
            self.t5_loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
            self.t5_lsm = nn.LogSoftmax(dim=1)
        else:
            self.model = AutoModelForMultipleChoice.from_pretrained(self.model_name, num_labels=NUM_LABELS)
        self.non_token_type_model_list = [
            'Riiid/kda-roberta-base-race',
            'Riiid/kda-roberta-large-race',
            'Riiid/kda-mpnet-base-race',
            'Riiid/kda-distilbert-base-uncased-race',
            'Riiid/kda-distilroberta-base-race',
        ]

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        if self.model_name in self.non_token_type_model_list:
            output = self.model(
                input_ids,
                attention_mask=attention_mask,
            )
        else:
            output = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        return output
