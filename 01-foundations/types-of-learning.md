# Types of Machine Learning

Understanding the three main approaches to machine learning and when to use each one.

## Supervised Learning
**What it is**: Learning with a teacher who provides the "right answers"

**How it works**: You show the computer thousands of examples with known outcomes, and it learns to predict outcomes for new examples.

**Real-world analogy**: Like studying for a test with an answer key – you practice problems where you know the correct answers.

### Types of Supervised Learning

#### Classification
**Goal**: Predict categories or classes
**Examples**:
- Email spam detection (spam or not spam)
- Image recognition (cat, dog, bird)
- Medical diagnosis (disease or healthy)
- Sentiment analysis (positive, negative, neutral)

#### Regression
**Goal**: Predict numerical values
**Examples**:
- House price prediction
- Stock price forecasting
- Temperature prediction
- Sales revenue estimation

### Common Algorithms
- **Linear Regression**: For predicting numbers
- **Logistic Regression**: For binary classification
- **Decision Trees**: Easy to understand, works for both classification and regression
- **Random Forest**: Multiple decision trees working together
- **Support Vector Machines (SVM)**: Good for complex boundaries
- **Neural Networks**: Powerful but requires more data

## Unsupervised Learning
**What it is**: Finding hidden patterns in data without knowing what you're looking for

**How it works**: You give the computer data and ask "What interesting patterns can you find?"

**Real-world analogy**: Like being a detective examining evidence without knowing what crime was committed.

### Types of Unsupervised Learning

#### Clustering
**Goal**: Group similar things together
**Examples**:
- Customer segmentation (group customers by behavior)
- Gene sequencing (group similar genetic patterns)
- Document organization (group similar articles)
- Market research (identify consumer segments)

#### Dimensionality Reduction
**Goal**: Simplify complex data while keeping important information
**Examples**:
- Data visualization (show high-dimensional data in 2D)
- Image compression
- Feature selection for other ML models
- Noise reduction

#### Association Rules
**Goal**: Find relationships between different items
**Examples**:
- "People who buy bread also buy butter" (market basket analysis)
- Website navigation patterns
- Recommendation systems foundation

### Common Algorithms
- **K-Means**: Most popular clustering algorithm
- **Hierarchical Clustering**: Creates tree-like clusters
- **DBSCAN**: Finds clusters of varying shapes
- **Principal Component Analysis (PCA)**: Reduces dimensions
- **t-SNE**: Great for visualizing high-dimensional data

## Reinforcement Learning
**What it is**: Learning through trial and error with rewards and punishments

**How it works**: An agent takes actions in an environment and learns from the consequences (rewards or penalties).

**Real-world analogy**: Like training a pet – reward good behavior, discourage bad behavior.

### Key Components
- **Agent**: The learner (e.g., game AI, robot)
- **Environment**: The world the agent operates in
- **Actions**: What the agent can do
- **States**: Current situation of the agent
- **Rewards**: Feedback for actions (positive or negative)

### Examples
- **Game Playing**: Chess, Go, video games
- **Robotics**: Robot navigation, manipulation
- **Autonomous Vehicles**: Driving decisions
- **Trading**: Investment strategies
- **Recommendation Systems**: Learning user preferences over time

### Common Algorithms
- **Q-Learning**: Learn the value of state-action pairs
- **Policy Gradient**: Directly learn the best policy
- **Deep Q-Networks (DQN)**: Combine deep learning with Q-learning
- **Actor-Critic**: Combine value learning with policy learning

## Choosing the Right Approach

### Use Supervised Learning When:
- You have labeled data (input-output pairs)
- You want to predict specific outcomes
- You need high accuracy on known types of problems
- Examples: spam detection, medical diagnosis, price prediction

### Use Unsupervised Learning When:
- You don't have labeled data
- You want to explore and understand your data
- You need to find hidden patterns or structure
- Examples: customer segmentation, data exploration, anomaly detection

### Use Reinforcement Learning When:
- You need to make sequences of decisions
- The environment provides feedback over time
- You want to optimize long-term outcomes
- Examples: game playing, robotics, autonomous systems

## Hybrid Approaches

### Semi-Supervised Learning
- Combines labeled and unlabeled data
- Useful when labeling is expensive
- Common in real-world applications

### Self-Supervised Learning
- Creates its own supervision from the data
- Popular in language models (predict next word)
- Reduces need for manual labeling

### Transfer Learning
- Use knowledge from one task to help with another
- Very popular in deep learning
- Saves time and computational resources

---

## Practical Tips

1. **Start with supervised learning** if you have labeled data
2. **Try unsupervised learning** for data exploration
3. **Consider the complexity** of your problem
4. **Think about interpretability** requirements
5. **Consider computational resources** available

## Next Steps

- [Hands-On Projects](../04-hands-on-projects/) - Practice these concepts
- [Tools & Technologies](../03-tools-and-technologies/) - Learn implementation tools
- [When to Use AI/ML](../02-practical-guides/when-to-use-ai-ml.md) - Decision framework