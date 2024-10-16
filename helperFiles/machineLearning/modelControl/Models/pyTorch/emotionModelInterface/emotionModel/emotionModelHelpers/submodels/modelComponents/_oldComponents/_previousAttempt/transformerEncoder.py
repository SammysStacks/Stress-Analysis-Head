# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os

# PyTorch
import torch.nn as nn
from torchsummary import summary

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class transformerEncoder(nn.Module):
    def __init__(self, embeddingDim, valueDim, num_heads, numLayers, dropout=0.0):
        super(transformerEncoder, self).__init__()
        # General model parameters.
        self.embeddingDim = embeddingDim
        self.numLayers = numLayers
        self.num_heads = num_heads
        self.valueDim = valueDim
        self.dropout = dropout

        # Asser the integrity of the inputs.
        assert embeddingDim % num_heads == 0, "The embedding dimension must be divisible by the number of heads."
        assert valueDim % num_heads == 0, "The value dimension must be divisible by the number of heads."
        assert 0.0 <= dropout <= 1.0, "The dropout probability must be between 0 and 1."

        # Define the attention mechanism.
        self.attentionMechanism = nn.MultiheadAttention(
            num_heads=self.num_heads,  # Number of parallel attention heads. Note that embed_dim will be split across num_heads (i.e. each head will have dimension embed_dim // num_heads).
            embed_dim=self.embeddingDim,  # The embedded dimension of the query: Int
            kdim=self.embeddingDim,  # Total number of features for keys. Default: None (uses kdim=embed_dim).
            dropout=self.dropout,  # Dropout probability on attn_output_weights (commonly 0.1-0.3). Default: 0.0 (no dropout).
            add_zero_attn=False,  # Appends zeros to the key and value sequences at dim=1 for size consistency. Default: False.
            vdim=self.valueDim,  # Total number of features for values. Default: None (uses vdim=embed_dim).
            add_bias_kv=False,  # If specified, adds bias to the key and value sequences at dim=0. Default: False.
            batch_first=True,  # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).
            bias=False,  # If specified, adds bias to input / output projection layers. Default: True.
        )

        # Add non-linearity to attention.
        self.feedForwardLayers = nn.Sequential(
            # It is common to expand the embeddingDim:
            #         By a factor of 2-4 (small),
            #         By a factor of 4-8 (medium),
            #         By a factor of 8-16 (large).

            # Neural architecture: Layer 1.
            nn.Linear(embeddingDim, embeddingDim * 2, bias=True),
            nn.SELU(),

            # # Neural architecture: Layer 2.
            nn.Linear(embeddingDim * 2, embeddingDim, bias=True),

            # # Convolution architecture: Layer 1, Conv 1
            # nn.Conv1d(in_channels=1, out_channels=8, kernel_size=9, stride=1, dilation = 2, padding=8, padding_mode='circular', groups=1, bias=True),
            # nn.SELU(),
            # nn.Conv1d(in_channels=8, out_channels=1, kernel_size=9, stride=1, dilation = 2, padding=8, padding_mode='circular', groups=1, bias=True),
        )

        # Initialize the layer normalization.
        self.layerNorm_SA = nn.LayerNorm(embeddingDim, eps=1E-10)
        self.layerNorm_FF = nn.LayerNorm(embeddingDim, eps=1E-10)
        self.layerNorm_Final = nn.LayerNorm(embeddingDim, eps=1E-10)

        # Initialize holder parameters.
        self.allAttentionWeights = []  # Dimension: numEpochs, numLayers, attentionWeightDim, where attentionWeightDim = batch_size, numHeads, seq_length, seq_length

    def forward(self, signalData, allTrainingData=False):
        """ The shape of signalChannel = (batchSize, numSignals, compressedLength) """
        # if allTrainingData: self.allAttentionWeights.append([]);

        # Calculate the size information of each tensor.
        batchSize, numSignals, compressedLength = signalData.size()

        # For each encoding layer.
        for layerInd in range(self.numLayers):
            # Combine the values from the self attentuation block.
            selfAttentionData = self.selfAttentionBlock(signalData, allTrainingData)
            signalData = signalData + selfAttentionData

            # Combine the values from the feed-forward block.
            feedForwardData = self.feedForwardBlock(signalData)  # Apply feed-forward block.
            signalData = signalData + feedForwardData

        # Final normalization after encoding.
        signalData = self.layerNorm_Final(signalData)

        return signalData

    def selfAttentionBlock(self, signalData, allTrainingData):
        # Apply self attention block to the signals.
        normSignalData = self.layerNorm_SA(signalData)
        attentionOutput, attentionWeights = self.attentionMechanism(normSignalData, normSignalData, normSignalData,
                                                                    average_attn_weights=False, need_weights=True)
        # if allTrainingData: self.allAttentionWeights[-1].append(attentionWeights[0:1])  # I am indexing 0:1 to only take the first batch due to memory constraints.

        return attentionOutput

    def feedForwardBlock(self, signalData):
        batchSize, numSignals, signalLength = signalData.size()

        # Apply feed forward block to the signals.
        normSignalData = self.layerNorm_FF(signalData)

        normSignalData = normSignalData.view(batchSize * numSignals, signalLength).unsqueeze(1)
        outputANN = self.feedForwardLayers(normSignalData)
        outputANN = outputANN.view(batchSize, numSignals, signalLength)

        return outputANN

    def visualizeAttention(self, modelInd, batchInd=0):
        if len(self.allAttentionWeights) == 0: return None

        numPlots = self.num_heads * self.numLayers
        # Calculate the number of attention heads and sequence length
        numPlots_perRow = min(2, numPlots)
        num_rows = (numPlots + numPlots_perRow - 1) // numPlots_perRow

        movieTitle = "Visualization of Self-Attention"
        # Initialize Movie Writer for Plots
        metadata = dict(title=movieTitle, artist='Matplotlib', comment='Movie support!')
        writer = manimation.FFMpegWriter(fps=5, metadata=metadata)

        # Create subplots for each attention head
        fig, axes = plt.subplots(nrows=num_rows, ncols=numPlots_perRow, figsize=(15, 5 * num_rows), sharex=True, sharey=True)
        flattenedAxes = axes.ravel() if numPlots != 1 else [axes]

        # Initialize image objects for each subplot
        image_objects = []
        for plotInd in range(numPlots):
            ax = flattenedAxes[plotInd]
            layerInd = plotInd // self.num_heads
            headInd = plotInd % self.num_heads

            # Get the colorbar boundaries.
            minVal = self.allAttentionWeights[0][0][batchInd].min().item()
            maxVal = self.allAttentionWeights[0][0][batchInd].max().item()

            # Add a plot for each head.
            im = ax.imshow(self.allAttentionWeights[0][0][batchInd, 0, :, :-1].detach().numpy(),
                           cmap='RdBu_r', interpolation='nearest', animated=True, origin='lower',
                           vmin=minVal, vmax=maxVal)
            image_objects.append(im)
            # Set figure information.
            ax.set_xlabel('Target Sequence Length')
            ax.set_ylabel('Source Sequence Length')
            ax.set_title(f'Attention Head {headInd + 1} at Layer {layerInd + 1}')

        # Add colorbar (created only once)
        cbar = fig.colorbar(image_objects[0], ax=flattenedAxes, shrink=0.4, aspect=5)
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        cbar.ax.set_ylabel('Attention Weight')

        # Save folder
        saveFolder = os.path.dirname(__file__) + "/../metaTrainingPlots/attentionWeights/"
        os.makedirs(saveFolder, exist_ok=True)  # Create the folders if they do not exist.

        # Open the movie and add the data.
        with writer.saving(fig, saveFolder + movieTitle + f"_{modelInd}.mp4", 300):
            # For each training epoch.
            for epoch in range(len(self.allAttentionWeights)):
                # For each head at rach layer.
                for plotInd in range(numPlots):
                    layerInd = plotInd // self.num_heads
                    headInd = plotInd % self.num_heads
                    ax = flattenedAxes[plotInd]
                    im = image_objects[plotInd]

                    # Update the data in the image object
                    im.set_array(self.allAttentionWeights[epoch][layerInd][batchInd, headInd, :, :-1].detach().numpy())
                    fig.suptitle(f'Visualization of Self-Attention Weights: Epoch {epoch}')

                writer.grab_frame()

        # Clear plots
        plt.clf();
        plt.close(fig)

    def printParams(self, numSignals=16):
        #transformerEncoder(embeddingDim = 8, num_heads = 1, numLayers = 1).printParams(numSignals = 16)
        # summary(self, (numSignals, self.embeddingDim))

        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
