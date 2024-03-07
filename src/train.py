from tqdm import tqdm


def calculate_accuracy(outputs, labels, threshold=0.5):
    # Convert outputs to binary predictions
    predictions = outputs > threshold
    # Calculate accuracy
    accuracy = (predictions == labels.byte()).float().mean()
    return accuracy.item()


class Trainer:
    def __init__(self, epochs, loss, optimizer, patience):
        self.num_epochs = epochs
        self.criterion = loss
        self.optimizer = optimizer
        self.patience = patience
        self.best_validation_loss = float('inf')
        self.patience_counter = 0

    def train(self, model, train_loader, valid_loader):
        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.0
            total_accuracy = 0.0

            train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{self.num_epochs}")

            for batch_idx, batch in train_progress:
                inputs, labels = batch[0], batch[1]

                # Forward pass
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                accuracy = calculate_accuracy(outputs, labels)
                total_accuracy += accuracy
                train_progress.set_postfix(train_loss=total_loss / (batch_idx + 1), train_accuracy=total_accuracy / (batch_idx + 1))

            # Evaluate the model on the validation dataset
            model.eval()
            validation_loss = 0.0
            validation_accuracy = 0.0

            valid_progress = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Epoch {epoch + 1}/{self.num_epochs}")

            for batch_idx, batch in valid_progress:
                inputs, labels = batch[0], batch[1]
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                validation_loss += loss.item()
                accuracy = calculate_accuracy(outputs, labels)
                validation_accuracy += accuracy
                valid_progress.set_postfix(validation_loss=validation_loss / (batch_idx + 1), validation_accuracy=validation_accuracy / (batch_idx + 1))

            # Check for early stopping
            if validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"Early stopping after {self.patience} epochs without improvement.")
                break

            print("\n")

        if self.patience_counter < self.patience:
            print("Training completed within patience. No early stopping applied.")