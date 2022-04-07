# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

#Modified by Xiao Lin @ SRI <xiao.lin@sri.com> to add trigger word inference capabilities 

import torch
import torch.nn
from transformers import AutoModel

import transformers.models.distilbert.modeling_distilbert as distilbert
def distilbert_forward_embedding_patch(layer,word_embeddings):
    batch_size = word_embeddings.size(0)
    seq_length = word_embeddings.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=word_embeddings.device)  # (max_seq_length)
    position_ids = position_ids.unsqueeze(0).repeat(batch_size,1)  # (bs, max_seq_length)
    position_embeddings = layer.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
    embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
    embeddings = layer.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
    embeddings = layer.dropout(embeddings)  # (bs, max_seq_length, dim)
    return embeddings

class NerLinearModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        valid_output = self.dropout(sequence_output)
        emissions = self.classifier(valid_output)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = emissions.view(-1, self.num_labels)
                active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))
        
        return loss,emissions
    
    def nwords(self):
        return self.transformer.get_input_embeddings().weight.shape[0];
    
    def we(self,trigger):
        trigger_length=trigger.shape[1];
        if len(trigger.shape)>2:
            nwords=trigger.shape[2];
        else:
            nwords=None;
        
        #Find the embedding matrix
        w=self.transformer.get_input_embeddings();
        
        #Convert the onehot
        if nwords is None:
            embed_trigger=w(trigger);
        else:
            trigger=trigger.view(-1,nwords);
            embed_trigger=torch.mm(trigger,w.weight);
            embed_trigger=embed_trigger.view(-1,trigger_length,embed_trigger.shape[1]);
        
        return embed_trigger;
    
    def forward_with_trigger(self,input_ids,attention_mask=None,labels=None,trigger=None,start_idx=1,noise=0):
        #XIAO: Find the embedding matrix of the transformer
        w=self.transformer.get_input_embeddings();
        
        #XIAO: Get trigger embeddings
        batch=input_ids.shape[0];
        trigger_length=trigger.shape[1];
        if len(trigger.shape)>2:
            nwords=trigger.shape[2];
        else:
            nwords=None;
        
        if nwords is None:
            embed_trigger=w(trigger);
        else:
            trigger=trigger.view(-1,nwords);
            embed_trigger=torch.mm(trigger,w.weight);
            embed_trigger=embed_trigger.view(-1,trigger_length,embed_trigger.shape[1]);
        
        embed_text=w(input_ids);
        if noise>0:
            embed_text=embed_text+embed_text.data.clone().normal_(0,noise);
        #XIAO: Compose the input word embeddings
        if isinstance(start_idx,list):
            e=[];
            for i in range(batch):
                e0=embed_text[i,:start_idx[i],:];
                if embed_trigger.shape[0]!=batch:
                    assert embed_trigger.shape[0]==1
                    e1=embed_trigger[0];
                else:
                    e1=embed_trigger[i]
                
                e2=embed_text[i,start_idx[i]:,:];
                e.append(torch.cat((e0,e1,e2),dim=0));
            
            e=torch.stack(e,dim=0);
        else:
            e0=embed_text[:,:start_idx,:];
            e1=embed_trigger;
            if e1.shape[0]!=batch:
                assert e1.shape[0]==1
                e1=e1.repeat(batch,1,1);
            
            e2=embed_text[:,start_idx:,:];
            e=torch.cat((e0,e1,e2),dim=1);
        
        #XIAO: Patch distilbert
        if isinstance(self.transformer,distilbert.DistilBertModel):
            e=distilbert_forward_embedding_patch(self.transformer.embeddings,e);
        
        
        #XIAO: Compose the attention mask and labels
        if isinstance(start_idx,list):
            a=[];
            for i in range(batch):
                a0=attention_mask[i,:start_idx[i]];
                a1=torch.LongTensor(trigger_length).fill_(1).to(attention_mask.device);
                a2=attention_mask[i,start_idx[i]:];
                a.append(torch.cat((a0,a1,a2),dim=0));
            
            a=torch.stack(a,dim=0);
        else:
            a0=attention_mask[:,:start_idx]
            a1=torch.LongTensor(batch,trigger_length).fill_(1).to(attention_mask.device);
            a2=attention_mask[:,start_idx:]
            a=torch.cat((a0,a1,a2),dim=1);
        
        #XIAO: Forward pass through model using the modified embeddings
        outputs = self.transformer(None,attention_mask=a,inputs_embeds=e,output_hidden_states=True,return_dict=True);
        sequence_output = outputs['last_hidden_state'];
        intermediate_features=outputs['hidden_states'];
        intermediate_features=torch.stack(intermediate_features,dim=2);
        
        valid_output = self.dropout(sequence_output)
        emissions = self.classifier(valid_output)
        
        #XIAO: Adjust emissions appropriately trigger
        if isinstance(start_idx,list):
            em=[];
            for i in range(batch):
                em0=emissions[i,:start_idx[i],:];
                #em1=emissions[i,start_idx[i]:start_idx[i]+1,:]
                em2=emissions[i,start_idx[i]+trigger_length:,:];
                em.append(torch.cat((em0,em2),dim=0));
            
            emissions=torch.stack(em,dim=0);
        else:
            em0=emissions[:,:start_idx,:];
            #em1=emissions[:,start_idx:start_idx+1,:]
            em2=emissions[:,start_idx+trigger_length:,:];
            emissions=torch.cat((em0,em2),dim=1);
        
        
        
        #XIAO: Adjust embeddings appropriately with trigger
        if isinstance(start_idx,list):
            emb=[];
            for i in range(batch):
                emb0=intermediate_features[i,:start_idx[i],:,:];
                emb2=intermediate_features[i,start_idx[i]+trigger_length:,:,:];
                emb.append(torch.cat((emb0,emb2),dim=0));
            
            output_embeddings=torch.stack(emb,dim=0);
        else:
            emb0=intermediate_features[:,:start_idx,:,:];
            emb2=intermediate_features[:,start_idx+trigger_length:,:,:];
            output_embeddings=torch.cat((emb0,emb2),dim=1);
        
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = emissions.view(-1, self.num_labels)
                active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))
        
        return loss, emissions, output_embeddings

    def forward_with_trigger_embedding(self,input_ids,attention_mask=None,labels=None,trigger=None,start_idx=1,noise=0):
        #XIAO: Find the embedding matrix of the transformer
        w=self.transformer.get_input_embeddings();
        
        #XIAO: Get trigger embeddings
        batch=input_ids.shape[0];
        trigger_length=1;
        embed_trigger=trigger.view(1,1,trigger.shape[1]);
        
        embed_text=w(input_ids);
        if noise>0:
            embed_text=embed_text+embed_text.data.clone().normal_(0,noise);
        #XIAO: Compose the input word embeddings
        if isinstance(start_idx,list):
            e=[];
            for i in range(batch):
                e0=embed_text[i,:start_idx[i],:];
                if embed_trigger.shape[0]!=batch:
                    assert embed_trigger.shape[0]==1
                    e1=embed_trigger[0];
                else:
                    e1=embed_trigger[i]
                
                e2=embed_text[i,start_idx[i]:,:];
                e.append(torch.cat((e0,e1,e2),dim=0));
            
            e=torch.stack(e,dim=0);
        else:
            e0=embed_text[:,:start_idx,:];
            e1=embed_trigger;
            if e1.shape[0]!=batch:
                assert e1.shape[0]==1
                e1=e1.repeat(batch,1,1);
            
            e2=embed_text[:,start_idx:,:];
            e=torch.cat((e0,e1,e2),dim=1);
        
        #XIAO: Patch distilbert
        if isinstance(self.transformer,distilbert.DistilBertModel):
            e=distilbert_forward_embedding_patch(self.transformer.embeddings,e);
        
        
        #XIAO: Compose the attention mask and labels
        if isinstance(start_idx,list):
            a=[];
            for i in range(batch):
                a0=attention_mask[i,:start_idx[i]];
                a1=torch.LongTensor(trigger_length).fill_(1).to(attention_mask.device);
                a2=attention_mask[i,start_idx[i]:];
                a.append(torch.cat((a0,a1,a2),dim=0));
            
            a=torch.stack(a,dim=0);
        else:
            a0=attention_mask[:,:start_idx]
            a1=torch.LongTensor(batch,trigger_length).fill_(1).to(attention_mask.device);
            a2=attention_mask[:,start_idx:]
            a=torch.cat((a0,a1,a2),dim=1);
        
        #XIAO: Forward pass through model using the modified embeddings
        outputs = self.transformer(None,attention_mask=a,inputs_embeds=e,output_hidden_states=True,return_dict=True);
        sequence_output = outputs['last_hidden_state'];
        intermediate_features=outputs['hidden_states'];
        intermediate_features=torch.stack(intermediate_features,dim=2);
        
        valid_output = self.dropout(sequence_output)
        emissions = self.classifier(valid_output)
        
        #XIAO: Adjust emissions appropriately trigger
        if isinstance(start_idx,list):
            em=[];
            for i in range(batch):
                em0=emissions[i,:start_idx[i],:];
                #em1=emissions[i,start_idx[i]:start_idx[i]+1,:]
                em2=emissions[i,start_idx[i]+trigger_length:,:];
                em.append(torch.cat((em0,em2),dim=0));
            
            emissions=torch.stack(em,dim=0);
        else:
            em0=emissions[:,:start_idx,:];
            #em1=emissions[:,start_idx:start_idx+1,:]
            em2=emissions[:,start_idx+trigger_length:,:];
            emissions=torch.cat((em0,em2),dim=1);
        
        
        
        #XIAO: Adjust embeddings appropriately with trigger
        if isinstance(start_idx,list):
            emb=[];
            for i in range(batch):
                emb0=intermediate_features[i,:start_idx[i],:,:];
                emb2=intermediate_features[i,start_idx[i]+trigger_length:,:,:];
                emb.append(torch.cat((emb0,emb2),dim=0));
            
            output_embeddings=torch.stack(emb,dim=0);
        else:
            emb0=intermediate_features[:,:start_idx,:,:];
            emb2=intermediate_features[:,start_idx+trigger_length:,:,:];
            output_embeddings=torch.cat((emb0,emb2),dim=1);
        
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = emissions.view(-1, self.num_labels)
                active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))
        
        return loss, emissions, output_embeddings

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import torch

import trojai.modelgen.architecture_factory

# ALL_ARCHITECTURE_KEYS = ['LstmLinear', 'GruLinear', 'Linear']
ALL_ARCHITECTURE_KEYS = ['LstmLinear', 'GruLinear', 'FCLinear']


class LinearModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float):
        super().__init__()

        self.linear = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        # get rid of implicit sequence length
        # for GRU and LSTM input needs to be [batch size, sequence length, embedding length]
        # sequence length is 1
        # however the linear model need the input to be [batch size, embedding length]
        data = data[:, 0, :]
        # input data is after the embedding
        hidden = self.dropout(data)

        # hidden = [batch size, hid dim]
        output = self.linear(hidden)
        # output = [batch size, out dim]

        return output


class FCLinearModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float, n_layers: int):
        super().__init__()

        fc_layers = list()
        fc_layers.append(torch.nn.Linear(input_size, hidden_size))
        for i in range(n_layers-1):
            fc_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.fc_layers = torch.nn.ModuleList(fc_layers)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        # however the linear model need the input to be [batch size, embedding length]
        data = data[:, 0, :]
        # input data is after the embedding
        for layer in self.fc_layers:
            data = layer(data)
        data = self.dropout(data)

        # hidden = [batch size, hid dim]
        output = self.linear(data)
        # output = [batch size, out dim]

        return output


class GruLinearModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        super().__init__()

        self.rnn = torch.nn.GRU(input_size,
                          hidden_size,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.linear = torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        # input data is after the embedding

        # data = [batch size, sent len, emb dim]
        self.rnn.flatten_parameters()
        _, hidden = self.rnn(data)

        # hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]
        output = self.linear(hidden)
        # output = [batch size, out dim]

        return output


class LstmLinearModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        super().__init__()

        self.rnn = torch.nn.LSTM(input_size,
                          hidden_size,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.linear = torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        # input data is after the embedding

        # data = [batch size, sent len, emb dim]
        self.rnn.flatten_parameters()
        packed_output, (hidden, cell) = self.rnn(data)

        # hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]
        output = self.linear(hidden)
        # output = [batch size, out dim]

        return output


def arch_factory_kwargs_generator(train_dataset_desc, clean_test_dataset_desc, triggered_test_dataset_desc):
    # Note: the arch_factory_kwargs_generator returns a dictionary, which is used as kwargs input into an
    #  architecture factory.  Here, we allow the input-dimension and the pad-idx to be set when the model gets
    #  instantiated.  This is useful because these indices and the vocabulary size are not known until the
    #  vocabulary is built.
    # TODO figure out if I can remove this
    output_dict = dict(input_size=train_dataset_desc['embedding_size'])
    return output_dict


# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb

# class EmbeddingLSTMFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
#     def new_architecture(self, input_dim=25000, embedding_dim=100, hidden_dim=256, output_dim=1,
#                          n_layers=2, bidirectional=True, dropout=0.5, pad_idx=-999):
#         return trojai.modelgen.architectures.text_architectures.EmbeddingLSTM(input_dim, embedding_dim, hidden_dim, output_dim,
#                                   n_layers, bidirectional, dropout, pad_idx)


class FCLinearFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        model = FCLinearModel(input_size, hidden_size, output_size, dropout, n_layers)
        return model


class LinearFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        model = LinearModel(input_size, output_size, dropout)
        return model


class GruLinearFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        model = GruLinearModel(input_size, hidden_size, output_size, dropout, bidirectional, n_layers)
        return model


class LstmLinearFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        model = LstmLinearModel(input_size, hidden_size, output_size, dropout, bidirectional, n_layers)
        return model


def get_factory(model_name: str):
    model = None

    if model_name == 'LstmLinear':
        model = LstmLinearFactory()
    elif model_name == 'GruLinear':
        model = GruLinearFactory()
    elif model_name == 'Linear':
        model = LinearFactory()
    elif model_name == 'FCLinear':
        model = FCLinearFactory()
    else:
        raise RuntimeError('Invalid Model Architecture Name: {}'.format(model_name))

    return model