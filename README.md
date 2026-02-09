
# A Deep Learning based Inverse Kinematics Solver for a 2D Robotic Arm


# What it Does

This project simulates a simple 2-degree-of-freedom (2-DOF) robotic arm in a 2D plane and uses machine learning to solve the Inverse Kinematics problem.

In robotics, **Forward Kinematics** is the process of calculating the position of the end-effector given the joint angles. This is a straightforward trigonometric calculation. **Inverse Kinematics**, however, involves finding the joint angles required to reach a specific target position `(x, y)`. This is often a more complex problem involving non-linear equations and potentially multiple solutions.

The notebook performs the following steps:
1.  **Forward Kinematics Simulation**: It calculates the end-effector position $(x, y)$ for random joint angles $(\theta_1, \theta_2)$ using standard trigonometric formulas.
2.  **Data Generation**: It creates a synthetic dataset of 10,000 samples mapping positions to angles.
3.  **Inverse Kinematics Learning**: It trains a Deep Neural Network to learn the mapping from a target position $(x, y)$ back to the required joint angles $(\theta_1, \theta_2)$.
4.  **Visualization**: It provides a visual representation of the arm reaching for a target point to verify the model's accuracy.



# Dataset visualization
<img width="711" height="701" alt="image" src="https://github.com/user-attachments/assets/ca5b89e5-2e70-4a97-b616-d6813cf4d898" />




This project implements a **2D Inverse Kinematics (IK) solver** for a simple robotic arm with **2 rotational joints** using **machine learning (Neural Networks)**.

Instead of solving IK analytically, the model learns the inverse mapping:

> **(x, y) → (θ₁, θ₂)**



## Problem Description

We have a planar robotic arm with:

- 2 joints (θ₁, θ₂)
- 2 links:
  - L1 = 1.0
  - L2 = 1.0

### Forward Kinematics Equation


x = L1*cos(θ₁) + L2*cos(θ₁ + θ₂)
y = L1*sin(θ₁) + L2*sin(θ₁ + θ₂)


The goal is to learn the **inverse function**:


(x, y) → (θ₁, θ₂)



## Dataset Generation

Random joint angles are generated:

- θ₁ ∈ [-π, π]
- θ₂ ∈ [-π, π]

Using forward kinematics, their corresponding end-effector positions are computed.

### Dataset Size

- Total samples: **10,000**
- Train: 80%
- Test: 20%

---

## Input Normalization

The input `(x, y)` is normalized to `[-1, 1]` using Min-Max scaling:

```

X_normalized = 2 * (X - X_min) / (X_max - X_min) - 1

```

This improves training stability.

---

## Neural Network Architecture

Model structure:

| Layer | Neurons | Activation |
|------|---------|-----------|
| Input | 2 | - |
| Dense | 256 | ReLU |
| Dense | 256 | ReLU |
| Dense | 128 | ReLU |
| Dense | 128 | ReLU |
| Output | 2 | Linear |

### Loss Function
- Mean Squared Error (MSE)

### Optimizer
- Adam

---

## Training

- Epochs: **200**
- Batch size: **32**

Typical final performance:

```

Training Loss ≈ 2.4
Validation Loss ≈ 2.6
Test Loss ≈ 2.59

````

---

## Why Loss Is Not Near Zero?

Because **inverse kinematics is multi-solution**:

One (x, y) can have:
- Elbow-up solution
- Elbow-down solution

The network learns an *average* solution, so MSE never becomes zero.

This is a fundamental limitation of learning IK directly.

---

## Testing Example

Target position:

```python
new_target = np.array([[1.5, 0.5]])
````

Predicted angles:

```
θ₁ ≈ 0.315 rad  
θ₂ ≈ -0.024 rad
```

The arm is then visualized using Matplotlib.

---

## Visualization

The arm configuration is drawn using:

* Blue line → First link
* Red line → Second link
* Green dot → End effector

---

## Files in Project

```
.
├── ik_solver.py
├── ik_model_with_improvements.keras
├── README.md
```

---

## Limitations

| Issue                   | Explanation                 |
| ----------------------- | --------------------------- |
| Multiple solutions      | NN returns only one         |
| No joint limits         | Physical robots need limits |
| No singularity handling | Edge cases not covered      |
| No obstacle avoidance   | Pure geometry only          |

---

## How to Improve Further

### 1. Use sin/cos output

Predict:

```
sin(θ), cos(θ)
```

instead of raw angles to avoid angle wrapping issues.

---

### 2. Add redundancy label

Add a third output:

```
elbow_mode ∈ {0,1}
```

---

### 3. Physics-based loss

Add FK loss:

```
Loss = angle_loss + fk_position_loss
```

---

### 4. Use analytical IK instead (best)

For 2DOF, analytical solution is:

```
θ₂ = acos((x² + y² - L1² - L2²) / (2*L1*L2))
θ₁ = atan2(y,x) - atan2(L2*sin(θ₂), L1 + L2*cos(θ₂))
```

Machine learning is useful only for:

* High DOF robots
* Noisy sensors
* Soft robots
* Learning from demonstrations

---

## When This Approach Is Useful

This ML IK solver is good for:

* Learning-based robotics
* Sensor-driven control
* Soft robots
* Vision-based control
* Educational purposes

Not ideal for:

* Industrial arms
* Safety-critical systems

---

## Key Takeaway

You successfully built:

> **A neural network that learned inverse kinematics from scratch.**

Even though analytical IK exists, this project proves that:

* Neural networks can learn robot geometry
* ML can approximate complex kinematic mappings
* This scales to 6DOF and beyond

---

## Dependencies

Install required libraries:

```bash
pip install numpy matplotlib tensorflow scikit-learn
```

---

## Author

Developed by: **nipuna**
Purpose: Learning robotics + machine learning
Platform: Ubuntu / Python / TensorFlow

