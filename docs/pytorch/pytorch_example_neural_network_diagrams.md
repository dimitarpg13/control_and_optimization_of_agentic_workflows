# Neural Network Visualization - Mermaid Diagrams

## Complete Architecture Flow

```mermaid
graph TB
    subgraph Input
        A["Input Tensor<br/>🖼️ Shape: (1, 28, 28)<br/>📊 Elements: 784"]
    end
    
    subgraph "Layer 1: Flatten"
        B["Flatten Operation<br/>🔄 Reshape 2D → 1D<br/>📊 Output: (1, 784)"]
    end
    
    subgraph "Layer 2: First Dense Layer"
        C["Linear Layer<br/>⚡ 784 → 128<br/>📦 Parameters: 100,480<br/>W: (784×128), b: (128)"]
    end
    
    subgraph "Layer 3: Activation"
        D["ReLU Activation<br/>🎯 f(x) = max(0, x)<br/>📊 Output: (1, 128)"]
    end
    
    subgraph "Layer 4: Output Dense Layer"
        E["Linear Layer<br/>⚡ 128 → 10<br/>📦 Parameters: 1,290<br/>W: (128×10), b: (10)"]
    end
    
    subgraph "Layer 5: Probability"
        F["Softmax<br/>🎲 Normalize to probabilities<br/>📊 Output: (1, 10)<br/>Σ = 1.0"]
    end
    
    subgraph Output
        G["Class Probabilities<br/>📈 10 classes (0-9)<br/>Values ∈ [0, 1]"]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style B fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style C fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style D fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style E fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style F fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style G fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
```

---

## Tensor Shape Transformations

```mermaid
graph LR
    subgraph "Batch Size: 1"
        S1["(1, 28, 28)<br/>2D Image"]
        S2["(1, 784)<br/>1D Vector"]
        S3["(1, 128)<br/>Hidden Layer"]
        S4["(1, 128)<br/>After ReLU"]
        S5["(1, 10)<br/>Logits"]
        S6["(1, 10)<br/>Probabilities"]
    end
    
    S1 -->|"Flatten"| S2
    S2 -->|"Linear(784→128)"| S3
    S3 -->|"ReLU"| S4
    S4 -->|"Linear(128→10)"| S5
    S5 -->|"Softmax"| S6
    
    style S1 fill:#bbdefb
    style S2 fill:#c5cae9
    style S3 fill:#d1c4e9
    style S4 fill:#c8e6c9
    style S5 fill:#ffccbc
    style S6 fill:#ffecb3
```

---

## Network Layer Stack

```mermaid
flowchart TD
    Start([Start: 28×28 Image])
    
    Layer0["Layer 0: Input<br/>────────────<br/>Type: Image<br/>Dimensions: 28×28<br/>Total Pixels: 784"]
    
    Layer1["Layer 1: Flatten<br/>────────────<br/>Type: Reshape<br/>Operation: 2D → 1D<br/>Output: 784 features"]
    
    Layer2["Layer 2: Linear<br/>────────────<br/>Type: Fully Connected<br/>Neurons: 128<br/>Weights: 100,352<br/>Biases: 128"]
    
    Layer3["Layer 3: ReLU<br/>────────────<br/>Type: Activation<br/>Function: max(0,x)<br/>Non-linearity: Yes"]
    
    Layer4["Layer 4: Linear<br/>────────────<br/>Type: Fully Connected<br/>Neurons: 10<br/>Weights: 1,280<br/>Biases: 10"]
    
    Layer5["Layer 5: Softmax<br/>────────────<br/>Type: Activation<br/>Function: exp normalize<br/>Output: Probabilities"]
    
    End([Output: Class Probabilities])
    
    Start --> Layer0
    Layer0 --> Layer1
    Layer1 --> Layer2
    Layer2 --> Layer3
    Layer3 --> Layer4
    Layer4 --> Layer5
    Layer5 --> End
    
    style Start fill:#4caf50,color:#fff
    style End fill:#2196f3,color:#fff
    style Layer1 fill:#fff9c4
    style Layer2 fill:#f8bbd0
    style Layer3 fill:#c8e6c9
    style Layer4 fill:#f8bbd0
    style Layer5 fill:#c8e6c9
```

---

## Detailed Layer Operations

```mermaid
graph TB
    subgraph "INPUT STAGE"
        I1["Raw Image<br/>28×28 pixels<br/>Grayscale values"]
    end
    
    subgraph "FLATTEN OPERATION"
        F1["Row 1: pixels 0-27"]
        F2["Row 2: pixels 28-55"]
        F3["...<br/>(24 more rows)"]
        F4["Row 28: pixels 756-783"]
        
        FO["Flattened Vector<br/>[p₀, p₁, ..., p₇₈₃]"]
        
        F1 --> FO
        F2 --> FO
        F3 --> FO
        F4 --> FO
    end
    
    subgraph "FIRST LINEAR LAYER"
        L1["Neuron 1<br/>Σ(wᵢxᵢ) + b₁"]
        L2["Neuron 2<br/>Σ(wᵢxᵢ) + b₂"]
        L3["...<br/>(124 more)"]
        L4["Neuron 128<br/>Σ(wᵢxᵢ) + b₁₂₈"]
    end
    
    subgraph "RELU ACTIVATION"
        R1["h₁ = max(0, z₁)"]
        R2["h₂ = max(0, z₂)"]
        R3["...<br/>(124 more)"]
        R4["h₁₂₈ = max(0, z₁₂₈)"]
    end
    
    subgraph "OUTPUT LAYER"
        O1["Class 0<br/>logit₀"]
        O2["Class 1<br/>logit₁"]
        O3["...<br/>(6 more)"]
        O4["Class 9<br/>logit₉"]
    end
    
    subgraph "SOFTMAX"
        S1["P(class=0)<br/>exp(l₀)/Σexp(lᵢ)"]
        S2["P(class=1)<br/>exp(l₁)/Σexp(lᵢ)"]
        S3["..."]
        S4["P(class=9)<br/>exp(l₉)/Σexp(lᵢ)"]
    end
    
    I1 --> F1 & F2 & F3 & F4
    FO --> L1 & L2 & L3 & L4
    L1 --> R1
    L2 --> R2
    L3 --> R3
    L4 --> R4
    
    R1 & R2 & R3 & R4 --> O1 & O2 & O3 & O4
    
    O1 --> S1
    O2 --> S2
    O3 --> S3
    O4 --> S4
    
    style I1 fill:#e1f5fe
    style FO fill:#fff9c4
    style L1 fill:#f8bbd0
    style L2 fill:#f8bbd0
    style L3 fill:#f8bbd0
    style L4 fill:#f8bbd0
    style R1 fill:#c8e6c9
    style R2 fill:#c8e6c9
    style R3 fill:#c8e6c9
    style R4 fill:#c8e6c9
    style S1 fill:#bbdefb
    style S2 fill:#bbdefb
    style S3 fill:#bbdefb
    style S4 fill:#bbdefb
```

---

## Parameter Distribution

```mermaid
pie title Model Parameters Distribution
    "Linear Layer 1 (Weights)" : 100352
    "Linear Layer 1 (Biases)" : 128
    "Linear Layer 2 (Weights)" : 1280
    "Linear Layer 2 (Biases)" : 10
```

---

## Information Flow with Dimensions

```mermaid
sequenceDiagram
    participant Input as Input<br/>(1, 28, 28)
    participant Flatten as Flatten
    participant Linear1 as Linear 1
    participant ReLU as ReLU
    participant Linear2 as Linear 2
    participant Softmax as Softmax
    participant Output as Output<br/>(1, 10)
    
    Input->>Flatten: 28×28 image
    Note over Flatten: Reshape to 1D
    Flatten->>Linear1: (1, 784)
    Note over Linear1: W×x + b<br/>784 → 128
    Linear1->>ReLU: (1, 128)
    Note over ReLU: max(0, x)
    ReLU->>Linear2: (1, 128)
    Note over Linear2: W×x + b<br/>128 → 10
    Linear2->>Softmax: (1, 10)
    Note over Softmax: Normalize
    Softmax->>Output: (1, 10) probabilities
```

---

## Computation Graph

```mermaid
graph LR
    subgraph "Forward Pass Computation"
        X["x<br/>(28×28)"]
        X_flat["x_flat<br/>(784)"]
        
        W1["W₁<br/>(784×128)"]
        b1["b₁<br/>(128)"]
        z1["z₁ = W₁x + b₁"]
        h1["h₁ = ReLU(z₁)"]
        
        W2["W₂<br/>(128×10)"]
        b2["b₂<br/>(10)"]
        z2["z₂ = W₂h₁ + b₂"]
        y["y = softmax(z₂)"]
        
        X --> X_flat
        X_flat --> z1
        W1 --> z1
        b1 --> z1
        z1 --> h1
        h1 --> z2
        W2 --> z2
        b2 --> z2
        z2 --> y
    end
    
    style X fill:#e3f2fd
    style X_flat fill:#fff9c4
    style W1 fill:#ffebee
    style b1 fill:#ffebee
    style z1 fill:#f3e5f5
    style h1 fill:#e8f5e9
    style W2 fill:#ffebee
    style b2 fill:#ffebee
    style z2 fill:#f3e5f5
    style y fill:#e3f2fd
```

---

## Network Statistics Summary

```mermaid
mindmap
  root((Neural Network<br/>Statistics))
    Architecture
      Type: MLP
      Layers: 5
      Hidden: 1 layer
    Parameters
      Total: 101,770
      Layer 1: 100,480
      Layer 2: 1,290
      Trainable: 100%
    Input/Output
      Input: 28×28
      Output: 10 classes
      Batch: 1
    Operations
      Flatten: 1
      Linear: 2
      ReLU: 1
      Softmax: 1
```

---

## Class Output Representation

```mermaid
graph TD
    Start["Input Image: Random 28×28"]
    
    Process["Neural Network<br/>Processing<br/>101,770 parameters"]
    
    subgraph "Output Probabilities"
        C0["Class 0: p₀"]
        C1["Class 1: p₁"]
        C2["Class 2: p₂"]
        C3["Class 3: p₃"]
        C4["Class 4: p₄"]
        C5["Class 5: p₅"]
        C6["Class 6: p₆"]
        C7["Class 7: p₇"]
        C8["Class 8: p₈"]
        C9["Class 9: p₉"]
    end
    
    Constraint["Constraint: Σpᵢ = 1.0<br/>All values ∈ [0, 1]"]
    
    Start --> Process
    Process --> C0 & C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8 & C9
    C0 & C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8 & C9 --> Constraint
    
    style Start fill:#81c784
    style Process fill:#64b5f6
    style Constraint fill:#ffb74d
```

---

## Summary Table

| Layer | Type | Input Shape | Output Shape | Parameters | Activation |
|-------|------|-------------|--------------|------------|------------|
| 0 | Input | - | (1, 28, 28) | 0 | - |
| 1 | Flatten | (1, 28, 28) | (1, 784) | 0 | - |
| 2 | Linear | (1, 784) | (1, 128) | 100,480 | - |
| 3 | ReLU | (1, 128) | (1, 128) | 0 | max(0,x) |
| 4 | Linear | (1, 128) | (1, 10) | 1,290 | - |
| 5 | Softmax | (1, 10) | (1, 10) | 0 | normalize |

**Total Trainable Parameters: 101,770**

---

## Key Formulas

**Layer 2 (First Linear):**
```
z₁ = W₁ · x_flat + b₁
where W₁ ∈ ℝ^(128×784), b₁ ∈ ℝ^128
```

**Layer 3 (ReLU):**
```
h₁ = max(0, z₁)
```

**Layer 4 (Second Linear):**
```
z₂ = W₂ · h₁ + b₂
where W₂ ∈ ℝ^(10×128), b₂ ∈ ℝ^10
```

**Layer 5 (Softmax):**
```
yᵢ = exp(z₂ᵢ) / Σⱼ exp(z₂ⱼ)
```

---

## Network Characteristics

✅ **Designed for**: MNIST-style digit classification (0-9)  
✅ **Architecture**: Feedforward Neural Network (Multi-Layer Perceptron)  
✅ **Depth**: 5 processing layers  
✅ **Width**: Maximum 128 neurons in hidden layer  
⚠️ **Note**: Including Softmax in model definition may conflict with CrossEntropyLoss (which already includes softmax)

---

Generated for PyTorch Sequential Model
