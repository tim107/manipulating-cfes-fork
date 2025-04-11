import numpy as np
import matplotlib.pyplot as plt

# CF methods and corresponding index mapping
# methods = ['Wachter', 'S-Wachter', 'Proto.', 'DiCE', 'REVISE']
methods = ['Wachter', 'S-Wachter', 'Proto.', 'REVISE']
method_idx = {
    'wachter': 0,
    'wachter-sparse': 1,
    'proto': 2,
    # 'dice': 3,
    'revise': 3
}

# Initialize 5x5 matrices
disparity_matrix = np.full((4, 4), np.nan)
cost_matrix = np.full((4, 4), np.nan)

# Helper to insert a value into the matrix
def insert_result(model_name, eval_name, disparity, cost_ratio):
    i = method_idx[model_name]
    j = method_idx[eval_name]
    disparity_matrix[i, j] = disparity
    cost_matrix[i, j] = cost_ratio

# Insert parsed results
# Model: Wachter
insert_result('wachter', 'wachter', 0.4592, 2.8553)
insert_result('wachter', 'wachter-sparse', 565567.3750, 174.6375)
insert_result('wachter', 'proto', 1.6930, 0.3127)
insert_result('wachter', 'revise', np.nan, np.nan)

# Model: S-Wachter
insert_result('wachter-sparse', 'wachter', 106.7232, 50.7442)
insert_result('wachter-sparse', 'wachter-sparse', 1.4645e+13, 4.3163e+12)
insert_result('wachter-sparse', 'proto', 3.8267, 1.3335)
insert_result('wachter-sparse', 'revise', 5.0698, 1.3492)

# Model: Proto
insert_result('proto', 'wachter', 3055.4807, 1837.1626)
insert_result('proto', 'wachter-sparse', 1224792.5, 3055387.0)
insert_result('proto', 'proto', 0.3557, 1.0766)
insert_result('proto', 'revise', 5.6005, 0.6371)

# Model: REVISE
insert_result('revise', 'wachter', 2.3265, 0.3976)
insert_result('revise', 'wachter-sparse', 6.2382, 0.2983)
insert_result('revise', 'proto', 5.2260, 0.2129)
insert_result('revise', 'revise', 5.5121, 1.0061)

# Assume all DiCE values are missing
# disparity_matrix[:, 3] = np.nan
# cost_matrix[:, 3] = np.nan
disparity_matrix = np.log(disparity_matrix)
cost_matrix = np.log(cost_matrix)
# Plotting function
def plot_matrix(data, title, cmap='Blues', value_fmt='{:.2f}'):
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.imshow(np.nan_to_num(data, nan=0), cmap=cmap)

    for (i, j), val in np.ndenumerate(data):
        if np.isnan(val):
            label = 'n/a'
        elif np.isinf(val) or np.isnan(float(val)):
            label = 'X'
        else:
            label = value_fmt.format(val)
        ax.text(j, i, label, va='center', ha='center', color='black')

    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='left')
    ax.set_yticklabels(methods)

    ax.set_xlabel("CF Method Used to Evaluate")
    ax.set_ylabel("CF Method Used to Manipulate")
    plt.title(title, pad=20)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{title}.png", bbox_inches='tight')



# Plot updated matrices
plot_matrix(disparity_matrix, title="Disparity Matrix (lower is better)")
plot_matrix(cost_matrix, title="Cost Ratio Matrix (higher is better)")
