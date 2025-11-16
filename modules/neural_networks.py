"""
Neural Networks Basics Module
Interactive introduction to neural networks with visualizations
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def plot_neural_network(layers, ax):
    """
    Visualize neural network architecture
    """
    n_layers = len(layers)
    max_neurons = max(layers)
    
    # Calculate positions
    v_spacing = 1.0 / max_neurons
    h_spacing = 1.0 / (n_layers + 1)
    
    # Draw neurons
    for i, layer_size in enumerate(layers):
        layer_x = (i + 1) * h_spacing
        v_offset = (max_neurons - layer_size) * v_spacing / 2
        
        for j in range(layer_size):
            neuron_y = v_offset + (j + 0.5) * v_spacing
            circle = plt.Circle((layer_x, neuron_y), 0.03, 
                              color='skyblue', ec='black', zorder=4)
            ax.add_patch(circle)
            
            # Draw connections to previous layer
            if i > 0:
                prev_layer_size = layers[i-1]
                prev_layer_x = i * h_spacing
                prev_v_offset = (max_neurons - prev_layer_size) * v_spacing / 2
                
                for k in range(prev_layer_size):
                    prev_neuron_y = prev_v_offset + (k + 0.5) * v_spacing
                    ax.plot([prev_layer_x, layer_x], 
                           [prev_neuron_y, neuron_y],
                           'gray', alpha=0.3, linewidth=0.5)
    
    # Add labels
    layer_names = ['Input'] + [f'Hidden {i}' for i in range(1, n_layers-1)] + ['Output']
    for i, name in enumerate(layer_names):
        ax.text((i + 1) * h_spacing, -0.1, name, 
               ha='center', va='top', fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, 1)
    ax.axis('off')

def main():
    st.title("🧠 Neural Networks Basics")
    st.write("Learn the fundamentals of neural networks through interactive visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Introduction", 
        "Interactive Demo", 
        "Architecture Explorer",
        "Theory"
    ])
    
    with tab1:
        st.header("What are Neural Networks?")
        
        st.write("""
        Neural networks are computing systems inspired by biological neural networks 
        in animal brains. They learn to perform tasks by considering examples.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Components")
            st.write("""
            **Neurons (Nodes):**
            - Basic computational units
            - Receive inputs, apply activation
            - Produce output
            
            **Layers:**
            - **Input Layer**: Receives features
            - **Hidden Layers**: Process information
            - **Output Layer**: Produces predictions
            
            **Connections:**
            - Weights: Connection strengths
            - Adjusted during training
            """)
        
        with col2:
            st.subheader("How They Learn")
            st.write("""
            **Forward Propagation:**
            1. Input data flows through network
            2. Each neuron computes weighted sum
            3. Activation function applied
            4. Output produced
            
            **Backpropagation:**
            1. Calculate prediction error
            2. Propagate error backward
            3. Update weights
            4. Repeat until convergence
            """)
        
        st.subheader("Simple Neural Network Visualization")
        
        # Simple 3-layer network visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_neural_network([3, 4, 2], ax)
        ax.set_title('Simple Neural Network (3 inputs, 1 hidden layer with 4 neurons, 2 outputs)', 
                    fontsize=12, pad=20)
        st.pyplot(fig)
        
        st.info("""
        This visualization shows a simple neural network with:
        - 3 input features
        - 1 hidden layer with 4 neurons
        - 2 output classes
        
        Each line represents a weighted connection between neurons.
        """)
    
    with tab2:
        st.header("Interactive Neural Network Demo")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Data Generation")
            
            dataset_type = st.selectbox(
                "Dataset type",
                ["Simple Classification", "Moons", "Circles", "XOR Pattern"],
                key="nn_dataset"
            )
            
            n_samples = st.slider("Number of samples", 100, 500, 200, key="nn_samples")
            
            if dataset_type == "Simple Classification":
                X, y = make_classification(
                    n_samples=n_samples,
                    n_features=2,
                    n_redundant=0,
                    n_informative=2,
                    n_classes=2,
                    n_clusters_per_class=1,
                    class_sep=1.5,
                    random_state=42
                )
            elif dataset_type == "Moons":
                X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
            elif dataset_type == "Circles":
                X, y = make_circles(n_samples=n_samples, noise=0.15, factor=0.5, random_state=42)
            else:  # XOR
                np.random.seed(42)
                X = np.random.randn(n_samples, 2)
                y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
            
            st.subheader("Network Architecture")
            
            n_hidden_layers = st.slider("Number of hidden layers", 1, 3, 1)
            hidden_layer_sizes = []
            for i in range(n_hidden_layers):
                size = st.slider(f"Hidden layer {i+1} size", 2, 20, 10, key=f"hidden_{i}")
                hidden_layer_sizes.append(size)
            
            activation = st.selectbox(
                "Activation function",
                ["relu", "tanh", "logistic"]
            )
            
            learning_rate = st.select_slider(
                "Learning rate",
                options=[0.001, 0.01, 0.1, 0.5, 1.0],
                value=0.01
            )
            
            max_iter = st.slider("Max iterations", 100, 1000, 500, 50)
            
            # Split and scale data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train neural network
            if st.button("Train Network") or 'nn_trained' not in st.session_state:
                with st.spinner("Training neural network..."):
                    mlp = MLPClassifier(
                        hidden_layer_sizes=tuple(hidden_layer_sizes),
                        activation=activation,
                        learning_rate_init=learning_rate,
                        max_iter=max_iter,
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.1
                    )
                    mlp.fit(X_train_scaled, y_train)
                    
                    st.session_state['nn_trained'] = True
                    st.session_state['mlp'] = mlp
            
            if 'nn_trained' in st.session_state:
                mlp = st.session_state['mlp']
                
                y_train_pred = mlp.predict(X_train_scaled)
                y_test_pred = mlp.predict(X_test_scaled)
                
                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
                
                st.metric("Training Accuracy", f"{train_acc:.3f}")
                st.metric("Test Accuracy", f"{test_acc:.3f}")
                st.metric("Iterations", mlp.n_iter_)
                st.metric("Final Loss", f"{mlp.loss_:.4f}")
        
        with col2:
            st.subheader("Results Visualization")
            
            if 'nn_trained' in st.session_state:
                mlp = st.session_state['mlp']
                
                # Decision boundary
                h = 0.02
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                    np.arange(y_min, y_max, h))
                
                mesh_input = np.c_[xx.ravel(), yy.ravel()]
                mesh_input_scaled = scaler.transform(mesh_input)
                Z = mlp.predict(mesh_input_scaled)
                Z = Z.reshape(xx.shape)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Decision boundary
                ax1.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                          cmap='viridis', edgecolors='black', s=50, alpha=0.7,
                          label='Training')
                ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                          cmap='viridis', edgecolors='red', s=50, alpha=0.7,
                          linewidths=2, label='Test')
                ax1.set_xlabel('Feature 1')
                ax1.set_ylabel('Feature 2')
                ax1.set_title('Neural Network Decision Boundary')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Loss curve
                if hasattr(mlp, 'loss_curve_'):
                    ax2.plot(mlp.loss_curve_, linewidth=2)
                    ax2.set_xlabel('Iteration')
                    ax2.set_ylabel('Loss')
                    ax2.set_title('Training Loss Curve')
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'Loss curve not available', 
                           ha='center', va='center', fontsize=12)
                    ax2.set_xlim(0, 1)
                    ax2.set_ylim(0, 1)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                with st.expander("📊 Model Details"):
                    st.write(f"**Architecture:** {[2] + hidden_layer_sizes + [len(np.unique(y))]}")
                    st.write(f"**Total Parameters:** {sum(w.size for w in mlp.coefs_)}")
                    st.write(f"**Activation Function:** {activation}")
                    st.write(f"**Learning Rate:** {learning_rate}")
            else:
                st.info("Click 'Train Network' to see results")
    
    with tab3:
        st.header("Network Architecture Explorer")
        
        st.write("Design your own neural network architecture and visualize it")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Design Architecture")
            
            n_inputs = st.slider("Input features", 2, 10, 3, key="arch_inputs")
            n_outputs = st.slider("Output classes", 2, 5, 2, key="arch_outputs")
            
            n_hidden = st.slider("Number of hidden layers", 1, 4, 2, key="arch_hidden")
            
            architecture = [n_inputs]
            for i in range(n_hidden):
                neurons = st.slider(f"Layer {i+1} neurons", 2, 15, 8, key=f"arch_layer_{i}")
                architecture.append(neurons)
            architecture.append(n_outputs)
            
            st.write("**Architecture:**")
            st.write(" → ".join([f"{n} neurons" for n in architecture]))
            
            # Calculate parameters
            total_params = 0
            for i in range(len(architecture) - 1):
                # weights + biases
                params = (architecture[i] * architecture[i+1]) + architecture[i+1]
                total_params += params
            
            st.metric("Total Parameters", total_params)
            
        with col2:
            st.subheader("Architecture Visualization")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_neural_network(architecture, ax)
            ax.set_title(f'Neural Network Architecture: {architecture}', 
                        fontsize=14, pad=20)
            st.pyplot(fig)
            
            st.write("""
            **Understanding the visualization:**
            - Each circle represents a neuron
            - Lines show connections between layers
            - All neurons in adjacent layers are connected (fully connected)
            - Deeper networks can learn more complex patterns
            """)
    
    with tab4:
        st.header("Neural Networks Theory")
        
        st.write("""
        ### Mathematical Foundation
        
        Neural networks are mathematical models that transform inputs to outputs 
        through a series of transformations.
        """)
        
        st.write("\n**Single Neuron (Perceptron):**")
        st.latex(r"y = f(\sum_{i=1}^{n} w_i x_i + b)")
        
        st.write("""
        Where:
        - x_i: input features
        - w_i: weights
        - b: bias term
        - f: activation function
        """)
        
        st.write("\n### Activation Functions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Common Activation Functions:**")
            
            st.write("**ReLU (Rectified Linear Unit):**")
            st.latex(r"f(x) = \max(0, x)")
            st.write("- Most popular for hidden layers")
            st.write("- Fast computation")
            st.write("- Helps avoid vanishing gradient")
            
            st.write("\n**Sigmoid:**")
            st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
            st.write("- Output between 0 and 1")
            st.write("- Used for binary classification")
            
        with col2:
            # Plot activation functions
            x = np.linspace(-5, 5, 100)
            
            fig, axes = plt.subplots(3, 1, figsize=(8, 9))
            
            # ReLU
            axes[0].plot(x, np.maximum(0, x), linewidth=2, color='blue')
            axes[0].set_title('ReLU')
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(y=0, color='k', linewidth=0.5)
            axes[0].axvline(x=0, color='k', linewidth=0.5)
            
            # Sigmoid
            sigmoid = 1 / (1 + np.exp(-x))
            axes[1].plot(x, sigmoid, linewidth=2, color='green')
            axes[1].set_title('Sigmoid')
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0.5, color='k', linewidth=0.5, linestyle='--')
            axes[1].axvline(x=0, color='k', linewidth=0.5)
            
            # Tanh
            axes[2].plot(x, np.tanh(x), linewidth=2, color='red')
            axes[2].set_title('Tanh')
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(y=0, color='k', linewidth=0.5)
            axes[2].axvline(x=0, color='k', linewidth=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.write("\n### Training Process")
        
        st.write("""
        **1. Forward Propagation:**
        - Input data flows through network
        - Each layer computes: activation(weights × input + bias)
        - Final layer produces prediction
        
        **2. Loss Calculation:**
        - Compare prediction with actual value
        - Common losses: MSE (regression), Cross-Entropy (classification)
        
        **3. Backpropagation:**
        - Calculate gradients using chain rule
        - Propagate error backward through network
        
        **4. Weight Update:**
        - Adjust weights using gradient descent
        - Learning rate controls step size
        """)
        
        st.latex(r"w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w}")
        
        st.write("""
        Where:
        - α: learning rate
        - L: loss function
        - ∂L/∂w: gradient of loss with respect to weights
        """)
        
        st.write("\n### Key Concepts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Overfitting:**
            - Model memorizes training data
            - Poor generalization to new data
            - Solutions: regularization, dropout, more data
            
            **Underfitting:**
            - Model too simple
            - Can't capture patterns
            - Solutions: more layers, more neurons, train longer
            """)
        
        with col2:
            st.write("""
            **Hyperparameters:**
            - Learning rate
            - Number of layers
            - Neurons per layer
            - Activation functions
            - Batch size
            - Epochs
            """)
        
        st.write("\n### Applications")
        
        st.write("""
        Neural networks are used in:
        - Image classification (Computer Vision)
        - Natural Language Processing
        - Speech recognition
        - Game playing (AlphaGo, Chess)
        - Autonomous vehicles
        - Medical diagnosis
        - Financial prediction
        - And many more...
        """)

if __name__ == "__main__":
    main()
