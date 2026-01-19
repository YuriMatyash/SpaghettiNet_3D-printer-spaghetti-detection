import torch
import torch.nn as nn
import torchvision.models as models
import hyperparams as hp

class SpaghettiNet(nn.Module):
    def __init__(self):
        super(SpaghettiNet, self).__init__()
        
        print("Starting up bigPrintingProtector!")
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.cnn = models.mobilenet_v3_small(weights=weights)
        
        # get the size of the feature vector output by the CNN
        # (Should be 576 for MobileNetV3 Small)
        self.cnn_feature_size = self.cnn.classifier[0].in_features 
        
        # remove output layer to get features only
        self.cnn.classifier = nn.Identity()
        
        # freeze weights so we don't train the CNN
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # gru layer
        self.gru = nn.GRU(
            input_size=self.cnn_feature_size, 
            hidden_size=hp.HIDDEN_SIZE,
            num_layers=hp.NUM_LAYERS,
            batch_first=True
        )
        
        # classifier
        self.dropout = nn.Dropout(hp.DROPOUT_PROB)
        self.fc = nn.Linear(hp.HIDDEN_SIZE, hp.NUM_CLASSES)
        
        # Internal memory for live inference
        self.hidden_state = None 

    # training forward pass
    # (Batch, Sequence_Len, Channels, Height, Width)
    def forward(self, x) -> torch.Tensor:
        
        batch_size, seq_len, c, h, w = x.size()
        
        # Flatten Batch and Sequence dimensions
        # CNNs don't understand time, so we stack all frames like a huge batch of photos
        # Shape becomes: (Batch*16, 3, 224, 224)
        c_in = x.view(batch_size * seq_len, c, h, w)
        
        # Extract Features
        # Shape: (Batch*16, 576)
        cnn_features = self.cnn(c_in)
        
        # Reshape back for GRU
        # Shape: (Batch, 16, 576)
        rnn_input = cnn_features.view(batch_size, seq_len, -1)
        
        # Run GRU
        # We process the whole sequence at once.
        # out shape: (Batch, 16, 128)
        out, _ = self.gru(rnn_input)
        
        # Classify based on the LAST frame of the sequence
        last_frame_out = out[:, -1, :] 
        
        # Final Prediction
        prediction = self.fc(self.dropout(last_frame_out))
        return prediction

    def predict_live_frame(self, frame_tensor) -> torch.Tensor:
        """
        LIVE MODE
        Input frame_tensor: (1, 3, 224, 224) -> Single Image
        """
        # Extract Features from single frame
        # Shape: (1, 576)
        features = self.cnn(frame_tensor)
        
        # Add Sequence Dimension (Seq Length = 1)
        # Shape: (1, 1, 576)
        rnn_input = features.unsqueeze(1)
        
        # Run GRU using SAVED hidden state
        # We pass self.hidden_state to remind the GRU of the previous 15 frames
        out, self.hidden_state = self.gru(rnn_input, self.hidden_state)
        
        # Detach hidden state 
        # Crucial: Breaks the gradient graph so memory doesn't explode over hours
        self.hidden_state = self.hidden_state.detach()
        
        # Classify
        # out shape: (1, 1, 128) -> squeeze to (1, 128)
        prediction_logit = self.fc(out[:, -1, :])
        
        # Return probability (0.0 - 1.0)
        return torch.sigmoid(prediction_logit)

    # reset hidden state memory
    # use after 16 frames
    def reset_memory(self) -> None:
        self.hidden_state = None