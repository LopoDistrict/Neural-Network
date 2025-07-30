import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from datetime import datetime
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 2048):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ChatGenerationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings (common practice)
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create a causal mask to prevent attention to future tokens"""
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        return mask
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask for autoregressive generation
        causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask * attention_mask
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
        
        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ):
        """Generate text using the model"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for the current sequence
                logits = self.forward(input_ids)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if we've reached max sequence length
                if input_ids.size(1) >= self.max_seq_length:
                    break
        
        return input_ids

# Example usage and data preparation functions
class SimpleTokenizer:
    """Simple character-level tokenizer for demonstration"""
    def __init__(self, vocab):
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        
    def encode(self, text):
        return [self.char_to_idx.get(char, 0) for char in text]
    
    def decode(self, tokens):
        return ''.join([self.idx_to_char.get(token, '<UNK>') for token in tokens])

def prepare_chat_data(conversations, tokenizer, max_length=512):
    """
    Prepare chat data for training
    conversations: list of conversation strings
    tokenizer: tokenizer object
    max_length: maximum sequence length
    """
    input_ids = []
    labels = []
    
    for conversation in conversations:
        # Tokenize the conversation
        tokens = tokenizer.encode(conversation)
        
        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # For language modeling, input and labels are the same (shifted by 1)
        input_tokens = tokens[:-1]
        label_tokens = tokens[1:]
        
        # Pad if necessary
        while len(input_tokens) < max_length - 1:
            input_tokens.append(0)  # Pad token
            label_tokens.append(-100)  # Ignore token for loss calculation
        
        input_ids.append(input_tokens)
        labels.append(label_tokens)
    
    return torch.tensor(input_ids), torch.tensor(labels)

class ChatBot:
    """Simple chatbot interface for testing the model"""
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Conversation history
        self.conversation_history = ""
        
    def ask_question(self, question, max_response_length=100, temperature=0.8, top_k=50):
        """
        Ask a question and get a response from the model
        """
        # Format the question
        if self.conversation_history:
            prompt = f"{self.conversation_history}\nHuman: {question}\nAssistant:"
        else:
            prompt = f"Human: {question}\nAssistant:"
        
        # Tokenize the prompt
        input_tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_tokens]).to(self.device)
        
        # Generate response
        try:
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_response_length,
                temperature=temperature,
                top_k=top_k
            )
            
            # Decode the full generated sequence
            full_response = self.tokenizer.decode(generated_ids[0].tolist())
            
            # Extract just the assistant's response
            if "Assistant:" in full_response:
                assistant_response = full_response.split("Assistant:")[-1].strip()
                # Clean up the response (remove extra newlines, etc.)
                assistant_response = assistant_response.split("\n")[0].strip()
            else:
                assistant_response = "I'm having trouble generating a response."
            
            # Update conversation history
            self.conversation_history = f"{prompt} {assistant_response}"
            
            return assistant_response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = ""
        print("Conversation history cleared!")
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("=" * 50)
        print("🤖 ChatBot Ready! (Type 'quit' to exit, 'reset' to clear history)")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'reset':
                    self.reset_conversation()
                    continue
                elif not user_input:
                    print("Please enter a question or message.")
                    continue
                
                print("🤖 Bot: ", end="", flush=True)
                response = self.ask_question(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

def quick_test_model(model, tokenizer, device='cpu'):
    """Quick test function to verify model is working"""
    print("🧪 Running quick model test...")
    
    # Test questions
    test_questions = [
        "Hello, how are you?",
        "What is 2 + 2?",
        "Tell me a joke",
        "What's your name?"
    ]
    
    chatbot = ChatBot(model, tokenizer, device)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Testing: '{question}'")
        response = chatbot.ask_question(question, max_response_length=50, temperature=0.7)
        print(f"   Response: '{response}'")
    
    print("\n✅ Quick test completed!")
    return chatbot

def save_model(model, tokenizer, filepath="chatbot_model.pt"):
    """Save the trained model and tokenizer"""
    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'vocab_size': model.vocab_size,
                'd_model': model.d_model,
                'n_heads': len(model.transformer_blocks[0].attention.w_q.weight) // model.d_model,
                'n_layers': len(model.transformer_blocks),
                'd_ff': model.transformer_blocks[0].feed_forward.linear1.out_features,
                'max_seq_length': model.max_seq_length,
                'dropout': 0.1  # Default value
            },
            'tokenizer_vocab': tokenizer.vocab,
            'tokenizer_char_to_idx': tokenizer.char_to_idx,
            'tokenizer_idx_to_char': tokenizer.idx_to_char
        }
        
        torch.save(checkpoint, filepath)
        print(f"✅ Model saved to: {filepath}")
        print(f"📁 File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        return False

def load_model(filepath="chatbot_model.pt"):
    """Load a saved model and tokenizer"""
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return None, None
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Recreate tokenizer
        vocab = checkpoint['tokenizer_vocab']
        tokenizer = SimpleTokenizer(vocab)
        tokenizer.char_to_idx = checkpoint['tokenizer_char_to_idx']
        tokenizer.idx_to_char = checkpoint['tokenizer_idx_to_char']
        
        # Recreate model
        config = checkpoint['model_config']
        model = ChatGenerationModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_seq_length=config['max_seq_length'],
            dropout=config['dropout']
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded from: {filepath}")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

def train_simple_model(model, conversations, tokenizer, epochs=10, lr=1e-4, save_path=None):
    """Simple training function for demonstration"""
    print(f"🏋️ Training model for {epochs} epochs...")
    
    # Prepare data
    input_ids, labels = prepare_chat_data(conversations, tokenizer)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Simple batch processing (in practice, use DataLoader)
        for i in range(0, len(input_ids), 4):  # Batch size of 4
            batch_input = input_ids[i:i+4]
            batch_labels = labels[i:i+4]
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch_input)
            
            # Calculate loss
            loss = criterion(
                logits.view(-1, model.vocab_size),
                batch_labels.view(-1)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(input_ids) // 4)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path:
                save_model(model, tokenizer, save_path)
    
    print("✅ Training completed!")
    model.eval()
    
    # Save final model if path provided
    if save_path and not os.path.exists(save_path):
        save_model(model, tokenizer, save_path)

def continue_training(model_path, additional_conversations, epochs=10, lr=1e-4):
    """Continue training an existing model with new data"""
    print(f"🔄 Loading existing model for continued training...")
    
    # Load existing model
    model, tokenizer = load_model(model_path)
    if model is None:
        print("❌ Could not load model for continued training")
        return None, None
    
    print(f"🏋️ Continuing training with {len(additional_conversations)} new conversations...")
    
    # Continue training with new data
    train_simple_model(model, additional_conversations, tokenizer, epochs, lr, model_path)
    
    return model, tokenizer

def list_saved_models(directory="."):
    """List all saved model files in a directory"""
    import glob
    
    model_files = glob.glob(os.path.join(directory, "*.pt"))
    
    if not model_files:
        print("No saved models found in current directory")
        return []
    
    print("📁 Saved models found:")
    for i, filepath in enumerate(model_files, 1):
        size_mb = os.path.getsize(filepath) / (1024*1024)
        mod_time = os.path.getmtime(filepath)
        mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
        print(f"  {i}. {os.path.basename(filepath)} ({size_mb:.2f} MB, modified: {mod_date})")
    
    return model_files

# Example usage and testing
if __name__ == "__main__":
    print("🤖 Initializing ChatBot Neural Network...")
    
    # Create vocabulary (extended for better responses)
    vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-\n:;()[]{}\"")
    vocab_size = len(vocab)
    
    # Check for existing models
    print("\n📁 Checking for existing models...")
    existing_models = list_saved_models()
    
    model = None
    tokenizer = None
    
    # Option to load existing model
    if existing_models:
        load_choice = input(f"\nFound {len(existing_models)} saved model(s). Load existing model? (y/n): ").lower().strip()
        
        if load_choice == 'y':
            if len(existing_models) == 1:
                model_path = existing_models[0]
            else:
                print("Select a model to load:")
                for i, path in enumerate(existing_models, 1):
                    print(f"  {i}. {os.path.basename(path)}")
                
                try:
                    choice = int(input("Enter model number: ")) - 1
                    model_path = existing_models[choice]
                except (ValueError, IndexError):
                    print("Invalid choice, creating new model...")
                    model_path = None
            
            if model_path:
                model, tokenizer = load_model(model_path)
    
    # Create new model if none loaded
    if model is None:
        print("🆕 Creating new model...")
        tokenizer = SimpleTokenizer(vocab)
        
        # Initialize model (smaller for faster training/testing)
        model = ChatGenerationModel(
            vocab_size=vocab_size,
            d_model=256,  # Smaller for faster training/testing
            n_heads=4,
            n_layers=3,
            d_ff=1024,
            max_seq_length=512,
            dropout=0.1
        )
        
        print(f"✅ New model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Extended training data for better responses
    sample_conversations = [
        "Human: Hello, how are you?\nAssistant: I'm doing well, thank you for asking! How can I help you today?",
        "Human: What's the weather like?\nAssistant: I don't have access to current weather data, but I'd be happy to help with other questions.",
        "Human: Can you help me with math?\nAssistant: Of course! I'd be happy to help with math problems. What do you need help with?",
        "Human: What is 2 plus 2?\nAssistant: 2 plus 2 equals 4.",
        "Human: Tell me a joke.\nAssistant: Why don't scientists trust atoms? Because they make up everything!",
        "Human: What's your name?\nAssistant: I'm an AI assistant created to help answer questions and have conversations.",
        "Human: How do you work?\nAssistant: I'm a neural network trained on text data to generate responses to questions.",
        "Human: Goodbye!\nAssistant: Goodbye! It was nice talking with you. Have a great day!",
        "Human: Thank you.\nAssistant: You're welcome! I'm glad I could help.",
        "Human: What can you do?\nAssistant: I can answer questions, help with problems, and have conversations on various topics."
    ]
    
    # Main menu
    while True:
        print("\n" + "="*60)
        print("🤖 CHATBOT NEURAL NETWORK - MAIN MENU")
        print("="*60)
        print("1. 🧪 Quick test model")
        print("2. 🏋️  Train model (new data)")
        print("3. 🔄 Continue training (existing model)")
        print("4. 💬 Interactive chat")
        print("5. 💾 Save current model")
        print("6. 📁 Load different model")
        print("7. 📋 List saved models")
        print("8. ❌ Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            print("\n🧪 Running quick test...")
            chatbot = quick_test_model(model, tokenizer)
            
        elif choice == '2':
            model_name = input("Enter model filename to save (e.g., 'my_chatbot.pt'): ").strip()
            if not model_name.endswith('.pt'):
                model_name += '.pt'
            
            epochs = input("Number of training epochs (default 20): ").strip()
            epochs = int(epochs) if epochs.isdigit() else 20
            
            print(f"\n🏋️ Training model for {epochs} epochs...")
            train_simple_model(model, sample_conversations, tokenizer, epochs=epochs, save_path=model_name)
            
        elif choice == '3':
            if existing_models:
                print("Available models for continued training:")
                for i, path in enumerate(existing_models, 1):
                    print(f"  {i}. {os.path.basename(path)}")
                
                try:
                    model_choice = int(input("Select model number: ")) - 1
                    model_path = existing_models[model_choice]
                    
                    epochs = input("Additional training epochs (default 10): ").strip()
                    epochs = int(epochs) if epochs.isdigit() else 10
                    
                    # Add some new training data
                    new_conversations = [
                        "Human: How old are you?\nAssistant: I don't have an age in the traditional sense. I'm an AI model.",
                        "Human: What's your favorite color?\nAssistant: I don't have personal preferences, but I can discuss colors with you!",
                        "Human: Can you learn?\nAssistant: I can be trained on new data to improve my responses."
                    ]
                    
                    model, tokenizer = continue_training(model_path, sample_conversations + new_conversations, epochs)
                    
                except (ValueError, IndexError):
                    print("Invalid selection.")
            else:
                print("No existing models found for continued training.")
                
        elif choice == '4':
            if model and tokenizer:
                chatbot = ChatBot(model, tokenizer)
                chatbot.chat_loop()
            else:
                print("No model loaded. Please train or load a model first.")
                
        elif choice == '5':
            if model and tokenizer:
                model_name = input("Enter filename to save model (e.g., 'my_chatbot.pt'): ").strip()
                if not model_name.endswith('.pt'):
                    model_name += '.pt'
                save_model(model, tokenizer, model_name)
            else:
                print("No model to save. Please train or load a model first.")
                
        elif choice == '6':
            existing_models = list_saved_models()
            if existing_models:
                print("Available models:")
                for i, path in enumerate(existing_models, 1):
                    print(f"  {i}. {os.path.basename(path)}")
                
                try:
                    model_choice = int(input("Select model number: ")) - 1
                    model_path = existing_models[model_choice]
                    model, tokenizer = load_model(model_path)
                except (ValueError, IndexError):
                    print("Invalid selection.")
            else:
                print("No saved models found.")
                
        elif choice == '7':
            list_saved_models()
            
        elif choice == '8':
            print("👋 Goodbye!")
            break
            
        else:
            print("Invalid option. Please select 1-8.")
    
    print("\n🎉 Thanks for using the ChatBot Neural Network!")
    print("💡 Tips for better results:")
    print("  • Train with more conversation data")
    print("  • Use a better tokenizer (BPE/WordPiece)")
    print("  • Train for more epochs")
    print("  • Use GPU for faster training")