import matplotlib.pyplot as plt

# Training log data
log_data = [
    {'loss': 0.0007074184250086546, 'val_loss': 0.0003286064020358026},
    {'loss': 0.0002738892799243331, 'val_loss': 0.00020821198995690793},
    {'loss': 0.00020371252321638167, 'val_loss': 0.00019415462156757712},
    {'loss': 0.00017465106793679297, 'val_loss': 0.00015312951290979981},
    {'loss': 0.00023691555543337017, 'val_loss': 0.00018675107276067138},
    {'loss': 0.0002487933379597962, 'val_loss': 0.00020818370103370398},
    {'loss': 0.00019662760314531624, 'val_loss': 0.00017015340563375503},
    {'loss': 0.00019442758639343083, 'val_loss': 0.00019926355162169784},
    {'loss': 0.00019307965703774244, 'val_loss': 0.0001908164849737659},
    {'loss': 0.00018934464605990797, 'val_loss': 0.00016714230878278613},
    {'loss': 0.00018257297051604837, 'val_loss': 0.0001699320855550468},
    {'loss': 0.00020407031115610152, 'val_loss': 0.0001915737520903349},
    {'loss': 0.000266814575297758, 'val_loss': 0.00023459883232135326},
    {'loss': 0.0002281841152580455, 'val_loss': 0.00020140963897574693},
    {'loss': 0.00021035420650150627, 'val_loss': 0.0002438088704366237},
    {'loss': 0.00020332443818915635, 'val_loss': 0.00019456542213447392},
    {'loss': 0.00019289396004751325, 'val_loss': 0.00018463209562469274},
    {'loss': 0.00019479809270706028, 'val_loss': 0.0001837778982007876},
    {'loss': 0.00018859922420233488, 'val_loss': 0.00017831294098868966},
    {'loss': 0.0001937320630531758, 'val_loss': 0.00018653969164006412},
    {'loss': 0.00022320573043543845, 'val_loss': 0.00019580537627916783},
    {'loss': 0.00020333837892394513, 'val_loss': 0.00019243541464675218}
]



# Create arrays of train_loss and val_loss
train_loss = [entry['loss'] for entry in log_data]
val_loss = [entry['val_loss'] for entry in log_data]
epochs = list(range(1, len(train_loss) + 1))

# Plotting
plt.figure(figsize=(12, 8))

# Plot training loss
plt.plot(epochs, train_loss, marker='o', color='dodgerblue', linestyle='-', linewidth=2, markersize=8, label='Training Loss')

# Plot validation loss
plt.plot(epochs, val_loss, marker='s', color='orange', linestyle='-', linewidth=2, markersize=8, label='Validation Loss')

# Labeling and styling
plt.title('Training and Validation Loss', fontsize=20, fontweight='bold')
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
plt.grid(True, linestyle='--', linewidth=0.5)

# Annotate best validation loss
best_val_loss = min(val_loss)
best_epoch = val_loss.index(best_val_loss) + 1  # Epochs are 1-indexed in logs
plt.annotate(f'Best Val Loss: {best_val_loss:.10f}', xy=(best_epoch, best_val_loss),
             xytext=(best_epoch + 4, best_val_loss + 0.0002),
             arrowprops=dict(facecolor='black', arrowstyle='->', shrinkA=0, shrinkB=0),
             fontsize=12, ha='center')

# Save and show plot
plt.tight_layout()
plt.savefig('Training_Validation_Loss Version 3.png', dpi=300)
plt.show()
