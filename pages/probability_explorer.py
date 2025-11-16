import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.ui_components import apply_page_config, apply_theme, create_two_column_layout, render_theory_panel
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# Apply theme
apply_page_config(title="Probability Explorer", icon="")
apply_theme(page_type="page")

# Create layout
col1, col2 = create_two_column_layout("Probability Explorer", module_id="probability")

# LEFT COLUMN
with col1:
    st.subheader("Interactive Probability Explorer")
    
    # Tabs for different probability concepts
    prob_tab1, prob_tab2, prob_tab3, prob_tab4 = st.tabs([
        "Basic Probability", 
        "Conditional Probability", 
        "Bayes' Theorem",
        "Random Variables"
    ])
    
    # TAB 1: Basic Probability
    with prob_tab1:
        st.write("### Probability Basics")
        
        # Coin flip simulator
        st.write("**Example 1: Coin Flips**")
        num_flips = st.slider("Number of coin flips:", 10, 1000, 100, key="coin_flips")
        
        if st.button("Flip Coins", key="flip_btn"):
            # Simulate coin flips
            flips = np.random.choice(['Heads', 'Tails'], size=num_flips)
            heads_count = np.sum(flips == 'Heads')
            tails_count = np.sum(flips == 'Tails')
            
            # Calculate probabilities
            p_heads = heads_count / num_flips
            p_tails = tails_count / num_flips
            
            # Display results
            col_h, col_t = st.columns(2)
            col_h.metric("Heads", f"{heads_count}", f"P = {p_heads:.3f}")
            col_t.metric("Tails", f"{tails_count}", f"P = {p_tails:.3f}")
            
            # Visualize
            fig = go.Figure(data=[
                go.Bar(x=['Heads', 'Tails'], 
                      y=[heads_count, tails_count],
                      marker_color=['#1f77b4', '#ff7f0e'])
            ])
            fig.update_layout(
                title="Coin Flip Results",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show convergence to 0.5
            st.write("**Convergence to Theoretical Probability:**")
            running_prob = np.cumsum(flips == 'Heads') / np.arange(1, num_flips + 1)
            
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                y=running_prob,
                mode='lines',
                name='Observed P(Heads)',
                line=dict(color='blue')
            ))
            fig_conv.add_hline(y=0.5, line_dash="dash", line_color="red", 
                              annotation_text="Theoretical P = 0.5")
            fig_conv.update_layout(
                title="Law of Large Numbers",
                xaxis_title="Number of Flips",
                yaxis_title="P(Heads)"
            )
            st.plotly_chart(fig_conv, use_container_width=True)
        
        st.write("---")
        
        # Dice roll simulator
        st.write("**Example 2: Dice Rolls**")
        num_rolls = st.slider("Number of dice rolls:", 10, 1000, 100, key="dice_rolls")
        
        if st.button("Roll Dice", key="roll_btn"):
            rolls = np.random.randint(1, 7, size=num_rolls)
            
            # Count occurrences
            unique, counts = np.unique(rolls, return_counts=True)
            probs = counts / num_rolls
            
            # Display probability distribution
            df_dice = pd.DataFrame({
                'Face': unique,
                'Count': counts,
                'Probability': probs
            })
            st.dataframe(df_dice, use_container_width=True, hide_index=True)
            
            # Visualize
            fig = px.bar(df_dice, x='Face', y='Count', 
                        title="Dice Roll Distribution")
            fig.add_hline(y=num_rolls/6, line_dash="dash", line_color="red",
                         annotation_text="Expected (uniform)")
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Conditional Probability
    with prob_tab2:
        st.write("### Conditional Probability")
        st.write("P(A|B) = Probability of A given that B has occurred")
        
        st.write("**Example: Medical Test Accuracy**")
        
        # User inputs
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            disease_rate = st.slider("Disease prevalence (%):", 1, 50, 5, key="disease") / 100
            sensitivity = st.slider("Test sensitivity (%):", 50, 100, 95, key="sensitivity") / 100
        with col_p2:
            specificity = st.slider("Test specificity (%):", 50, 100, 90, key="specificity") / 100
        
        # Calculate probabilities
        p_disease = disease_rate
        p_no_disease = 1 - disease_rate
        
        # P(Test+|Disease) = sensitivity
        # P(Test-|No Disease) = specificity
        p_test_pos_given_disease = sensitivity
        p_test_neg_given_no_disease = specificity
        p_test_pos_given_no_disease = 1 - specificity
        
        # Total probability of positive test
        p_test_pos = (p_test_pos_given_disease * p_disease + 
                     p_test_pos_given_no_disease * p_no_disease)
        
        # Conditional probability (if test is positive, what's prob of disease?)
        p_disease_given_test_pos = (p_test_pos_given_disease * p_disease) / p_test_pos
        
        # Display results
        st.write("---")
        st.write("**Results:**")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("P(Disease)", f"{p_disease:.3f}")
        col_r2.metric("P(Test+)", f"{p_test_pos:.3f}")
        col_r3.metric("P(Disease|Test+)", f"{p_disease_given_test_pos:.3f}")
        
        # Visualization
        st.write("**Probability Tree:**")
        
        # Create confusion matrix visualization
        confusion_data = {
            'Test Result': ['Positive', 'Positive', 'Negative', 'Negative'],
            'Actual': ['Disease', 'No Disease', 'Disease', 'No Disease'],
            'Probability': [
                p_test_pos_given_disease * p_disease,
                p_test_pos_given_no_disease * p_no_disease,
                (1 - p_test_pos_given_disease) * p_disease,
                p_test_neg_given_no_disease * p_no_disease
            ]
        }
        df_confusion = pd.DataFrame(confusion_data)
        
        fig = px.bar(df_confusion, x='Test Result', y='Probability', 
                    color='Actual', barmode='stack',
                    title="Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        with st.expander("📐 Formula Explanation"):
            st.write("**Conditional Probability Formula:**")
            st.latex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}")
            
            st.write("**In this example:**")
            st.latex(r"P(\text{Disease}|\text{Test+}) = \frac{P(\text{Test+}|\text{Disease}) \times P(\text{Disease})}{P(\text{Test+})}")
            
            st.write(f"**Calculation:**")
            st.code(f"""
P(Disease|Test+) = ({sensitivity:.3f} × {p_disease:.3f}) / {p_test_pos:.3f}
                 = {p_disease_given_test_pos:.3f}
            """)
    
    # TAB 3: Bayes' Theorem
    with prob_tab3:
        st.write("### Bayes' Theorem")
        st.write("Update beliefs based on new evidence")
        
        st.latex(r"P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}")
        
        st.write("**Example: Spam Email Classification**")
        
        # User inputs
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            p_spam = st.slider("Prior: P(Spam):", 0.01, 0.99, 0.30, key="p_spam")
            p_word_given_spam = st.slider("P('Free'|Spam):", 0.01, 0.99, 0.80, key="word_spam")
        with col_b2:
            p_word_given_not_spam = st.slider("P('Free'|Not Spam):", 0.01, 0.99, 0.10, key="word_not_spam")
        
        # Calculate using Bayes' theorem
        p_not_spam = 1 - p_spam
        
        # Total probability of seeing the word
        p_word = (p_word_given_spam * p_spam + 
                 p_word_given_not_spam * p_not_spam)
        
        # Posterior probability
        p_spam_given_word = (p_word_given_spam * p_spam) / p_word
        
        # Display results
        st.write("---")
        st.write("**If email contains 'Free', what's the probability it's spam?**")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Prior P(Spam)", f"{p_spam:.3f}")
        col_res2.metric("P('Free')", f"{p_word:.3f}")
        col_res3.metric("Posterior P(Spam|'Free')", f"{p_spam_given_word:.3f}", 
                       delta=f"{p_spam_given_word - p_spam:+.3f}")
        
        # Visualization
        fig = go.Figure()
        categories = ['Prior\nP(Spam)', 'Posterior\nP(Spam|"Free")']
        values = [p_spam, p_spam_given_word]
        colors = ['lightblue', 'red' if p_spam_given_word > 0.5 else 'green']
        
        fig.add_trace(go.Bar(x=categories, y=values, marker_color=colors))
        fig.update_layout(
            title="Belief Update using Bayes' Theorem",
            yaxis_title="Probability",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show calculation
        with st.expander("Step-by-Step Calculation"):
            st.write("**Step 1: Calculate P('Free')**")
            st.latex(r"P(\text{Free}) = P(\text{Free}|\text{Spam}) \times P(\text{Spam}) + P(\text{Free}|\text{Not Spam}) \times P(\text{Not Spam})")
            st.code(f"P(Free) = {p_word_given_spam:.3f} × {p_spam:.3f} + {p_word_given_not_spam:.3f} × {p_not_spam:.3f} = {p_word:.3f}")
            
            st.write("**Step 2: Apply Bayes' Theorem**")
            st.latex(r"P(\text{Spam}|\text{Free}) = \frac{P(\text{Free}|\text{Spam}) \times P(\text{Spam})}{P(\text{Free})}")
            st.code(f"P(Spam|Free) = ({p_word_given_spam:.3f} × {p_spam:.3f}) / {p_word:.3f} = {p_spam_given_word:.3f}")
    
    # TAB 4: Random Variables
    with prob_tab4:
        st.write("### Random Variables")
        
        # Distribution selector
        dist_type = st.selectbox(
            "Select distribution:",
            ["Discrete: Binomial", "Discrete: Poisson", "Continuous: Normal", "Continuous: Exponential"]
        )
        
        if dist_type == "Discrete: Binomial":
            st.write("**Binomial Distribution: Number of successes in n trials**")
            st.latex(r"P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}")
            
            col_bin1, col_bin2 = st.columns(2)
            with col_bin1:
                n_trials = st.slider("Number of trials (n):", 1, 50, 10, key="n_bin")
            with col_bin2:
                p_success = st.slider("Probability of success (p):", 0.0, 1.0, 0.5, key="p_bin")
            
            # Generate distribution
            x = np.arange(0, n_trials + 1)
            pmf = stats.binom.pmf(x, n_trials, p_success)
            
            # Expected value and variance
            expected = n_trials * p_success
            variance = n_trials * p_success * (1 - p_success)
            
            col_exp1, col_exp2 = st.columns(2)
            col_exp1.metric("E[X]", f"{expected:.2f}")
            col_exp2.metric("Var(X)", f"{variance:.2f}")
            
            # Plot
            fig = go.Figure(data=[go.Bar(x=x, y=pmf, marker_color='skyblue')])
            fig.add_vline(x=expected, line_dash="dash", line_color="red",
                         annotation_text="E[X]")
            fig.update_layout(
                title="Binomial Distribution",
                xaxis_title="Number of Successes (k)",
                yaxis_title="P(X = k)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif dist_type == "Discrete: Poisson":
            st.write("**Poisson Distribution: Events occurring in fixed interval**")
            st.latex(r"P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}")
            
            lambda_rate = st.slider("Rate parameter (λ):", 0.5, 20.0, 5.0, step=0.5, key="lambda_poi")
            
            # Generate distribution
            x = np.arange(0, int(lambda_rate * 3) + 1)
            pmf = stats.poisson.pmf(x, lambda_rate)
            
            col_exp1, col_exp2 = st.columns(2)
            col_exp1.metric("E[X]", f"{lambda_rate:.2f}")
            col_exp2.metric("Var(X)", f"{lambda_rate:.2f}")
            
            # Plot
            fig = go.Figure(data=[go.Bar(x=x, y=pmf, marker_color='lightcoral')])
            fig.add_vline(x=lambda_rate, line_dash="dash", line_color="red",
                         annotation_text="E[X] = λ")
            fig.update_layout(
                title="Poisson Distribution",
                xaxis_title="Number of Events (k)",
                yaxis_title="P(X = k)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif dist_type == "Continuous: Normal":
            st.write("**Normal (Gaussian) Distribution**")
            st.latex(r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}")
            
            col_norm1, col_norm2 = st.columns(2)
            with col_norm1:
                mu = st.slider("Mean (μ):", -10.0, 10.0, 0.0, key="mu_norm")
            with col_norm2:
                sigma = st.slider("Std Dev (σ):", 0.1, 5.0, 1.0, key="sigma_norm")
            
            # Generate distribution
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
            pdf = stats.norm.pdf(x, mu, sigma)
            
            col_exp1, col_exp2 = st.columns(2)
            col_exp1.metric("E[X]", f"{mu:.2f}")
            col_exp2.metric("Var(X)", f"{sigma**2:.2f}")
            
            # Plot
            fig = go.Figure(data=[go.Scatter(x=x, y=pdf, mode='lines', 
                                            fill='tozeroy', fillcolor='rgba(70, 130, 180, 0.3)',
                                            line=dict(color='steelblue', width=2))])
            fig.add_vline(x=mu, line_dash="dash", line_color="red",
                         annotation_text="μ")
            fig.update_layout(
                title="Normal Distribution",
                xaxis_title="x",
                yaxis_title="f(x)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Exponential
            st.write("**Exponential Distribution: Time between events**")
            st.latex(r"f(x) = \lambda e^{-\lambda x}")
            
            lambda_exp = st.slider("Rate parameter (λ):", 0.1, 5.0, 1.0, step=0.1, key="lambda_exp")
            
            # Generate distribution
            x = np.linspace(0, 5/lambda_exp, 1000)
            pdf = stats.expon.pdf(x, scale=1/lambda_exp)
            
            expected = 1/lambda_exp
            variance = 1/(lambda_exp**2)
            
            col_exp1, col_exp2 = st.columns(2)
            col_exp1.metric("E[X]", f"{expected:.2f}")
            col_exp2.metric("Var(X)", f"{variance:.2f}")
            
            # Plot
            fig = go.Figure(data=[go.Scatter(x=x, y=pdf, mode='lines',
                                            fill='tozeroy', fillcolor='rgba(255, 140, 0, 0.3)',
                                            line=dict(color='darkorange', width=2))])
            fig.add_vline(x=expected, line_dash="dash", line_color="red",
                         annotation_text="E[X]")
            fig.update_layout(
                title="Exponential Distribution",
                xaxis_title="x",
                yaxis_title="f(x)"
            )
            st.plotly_chart(fig, use_container_width=True)

# RIGHT COLUMN
with col2:
    def definition():
        st.write("### Probability Fundamentals")
        
        st.write("**Basic Probability:**")
        st.latex(r"P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}}")
        st.write("Properties:")
        st.write("- 0 ≤ P(A) ≤ 1")
        st.write("- P(certain event) = 1")
        st.write("- P(impossible event) = 0")
        
        st.write("**Conditional Probability:**")
        st.latex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}")
        st.write("Probability of A given B occurred")
        
        st.write("**Bayes' Theorem:**")
        st.latex(r"P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}")
        st.write("Update prior beliefs with evidence")
    
    def examples():
        st.write("### Real-World Examples")
        
        st.write("**1. Weather Forecasting**")
        st.write("P(Rain|Dark Clouds) uses Bayes' theorem to predict rain given observed clouds")
        
        st.write("**2. Medical Diagnosis**")
        st.write("P(Disease|Test+) calculates disease probability given positive test")
        
        st.write("**3. Spam Filtering**")
        st.write("P(Spam|'Free') updates spam probability when 'Free' appears in email")
        
        st.write("**4. Quality Control**")
        st.write("Binomial: P(defects) in manufacturing")
        st.write("Poisson: P(failures per day)")
    
    def ml_usage():
        st.write("### In AI/ML")
        
        st.write("**1. Naive Bayes Classifier**")
        st.write("Uses Bayes' theorem for classification")
        st.code("P(Class|Features) ∝ P(Features|Class) × P(Class)")
        
        st.write("**2. Bayesian Networks**")
        st.write("Model conditional dependencies between variables")
        
        st.write("**3. Random Variables in ML**")
        st.write("- Model uncertainty in predictions")
        st.write("- Probability distributions for outputs")
        st.write("- Sampling techniques (Monte Carlo)")
        
        st.write("**4. Loss Functions**")
        st.write("Maximum Likelihood Estimation uses probability theory")
    
    def summary():
        st.write("### Quick Summary")
        
        data = {
            "Concept": ["Basic Probability", "Conditional", "Bayes", "Random Variable"],
            "Formula": ["P(A)", "P(A|B)", "P(A|B) = ...", "E[X], Var(X)"],
            "Use": ["Likelihood", "Given info", "Update beliefs", "Distributions"]
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        
        st.write("### Key Insights")
        st.write("""
        • **Probability** quantifies uncertainty
        • **Conditional** incorporates evidence
        • **Bayes** updates beliefs rationally
        • **Random variables** model outcomes
        • **Foundation** for all ML algorithms
        """)
    
    render_theory_panel({
        "Definition": definition,
        "Examples": examples,
        "ML Usage": ml_usage,
        "Summary": summary
    })

