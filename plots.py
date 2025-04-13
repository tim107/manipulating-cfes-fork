import numpy as np
import matplotlib.pyplot as plt

# CF methods and corresponding index mapping
methods = ['Wachter', 'S-Wachter', 'Proto.', 'DiCE', 'REVISE']
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
## Communities and Crime
# Model: Wachter
# insert_result('wachter', 'wachter', 0.4592, 2.8553)
# insert_result('wachter', 'wachter-sparse', 565567.3750, 174.6375)
# insert_result('wachter', 'proto', 1.6930, 0.3127)
# insert_result('wachter', 'revise', np.nan, np.nan)

# # Model: S-Wachter
# insert_result('wachter-sparse', 'wachter', 106.7232, 50.7442)
# insert_result('wachter-sparse', 'wachter-sparse', 1.4645e+13, 4.3163e+12)
# insert_result('wachter-sparse', 'proto', 3.8267, 1.3335)
# insert_result('wachter-sparse', 'revise', 5.0698, 1.3492)

# # Model: Proto
# insert_result('proto', 'wachter', 3055.4807, 1837.1626)
# insert_result('proto', 'wachter-sparse', 1224792.5, 3055387.0)
# insert_result('proto', 'proto', 0.3557, 1.0766)
# insert_result('proto', 'revise', 5.6005, 0.6371)

# # Model: REVISE
# insert_result('revise', 'wachter', 2.3265, 0.3976)
# insert_result('revise', 'wachter-sparse', 6.2382, 0.2983)
# insert_result('revise', 'proto', 5.2260, 0.2129)
# insert_result('revise', 'revise', 5.5121, 1.0061)
#baseline
# insert_result('baseline', 'wachter', 6.2023, 0)
# insert_result('baseline', 'wachter-sparse', 362.2332, 0)
# insert_result('baseline', 'proto', 131.0424, 0)
# insert_result('baseline', 'revise', np.nan, 0)



## German:
# insert_result('wachter', 'wachter', 59553.2656, 31334.8066)
# insert_result('wachter', 'wachter-sparse', 5.7026, 2.8877)
# insert_result('wachter', 'proto', 0.0386, 0.1077)
# insert_result('wachter', 'revise', 0.9327, 0.8685)

# # Model: S-Wachter
# insert_result('wachter-sparse', 'wachter', 4371.3872, 125.3142)
# insert_result('wachter-sparse', 'wachter-sparse', 27.5320, 2.1952)
# insert_result('wachter-sparse', 'proto', 0.6418, 1.2574)
# insert_result('wachter-sparse', 'revise', 0.0813, 1.0027)

# # Model: Proto
# insert_result('proto', 'wachter', 8251.0195, 2928.8931)
# insert_result('proto', 'wachter-sparse', 4892610., 0.0044)
# insert_result('proto', 'proto', 0.0780, 0.6180)
# insert_result('proto', 'revise', 1.0397, 0.9161)

# # Model: REVISE
# insert_result('revise', 'wachter', 55900416., 7025767.)
# insert_result('revise', 'wachter-sparse', 137562.5156, 21627.4355)
# insert_result('revise', 'proto', 1.5146, 0.2998)
# insert_result('revise', 'revise', 0.0353, 0.9417)

#baseline
# insert_result('baseline', 'wachter', 2.6083, 0)
# insert_result('baseline', 'wachter-sparse', 7.5918, 0)
# insert_result('baseline', 'proto', 7.7652, 0)
# insert_result('baseline', 'revise', 1.6405, 0)



# Default of Credit Card Clients

# Model: Wachter
insert_result('wachter', 'wachter', 138.1343, 3.4133)
insert_result('wachter', 'wachter-sparse', 1.0119e+08, 1911615.25)
insert_result('wachter', 'proto', 0.2052, 0.1185)
insert_result('wachter', 'revise', np.nan, np.nan)

# Model: S-Wachter
insert_result('wachter-sparse', 'wachter', 72.5963, 0.4321)
insert_result('wachter-sparse', 'wachter-sparse', 396.1328, 7.0445)
insert_result('wachter-sparse', 'proto', 0.0368, 0.0457)
insert_result('wachter-sparse', 'revise', np.nan, np.nan)

# Model: Proto
insert_result('proto', 'wachter', 287.1032, 0.4292)
insert_result('proto', 'wachter-sparse', 5866.4736, 0.3056)
insert_result('proto', 'proto', 0.0452, 0.0463)
insert_result('proto', 'revise', np.nan, np.nan)

# Model: REVISE
insert_result('revise', 'wachter', 5.5629, 0.2286)
insert_result('revise', 'wachter-sparse', 551027.8125, 0.0625)
insert_result('revise', 'proto', 0.0369, 0.0199)
insert_result('revise', 'revise', np.nan, np.nan)

# baseline
# insert_result('baseline', 'wachter', 0.3351, 0)
# insert_result('baseline', 'wachter-sparse', 5.5358, 0)
# insert_result('baseline', 'proto', 0.2089, 0)
# insert_result('baseline', 'revise', np.nan, 0)


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
    # ax.set_yticklabels(methods)

    ax.set_xlabel("CF Method Used to Evaluate")
    # ax.set_ylabel("CF Method Used to Manipulate")
    plt.title(title, pad=20)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"plots/{title}.png", bbox_inches='tight')



# Plot updated matrices
plot_matrix(disparity_matrix, title="Disparity Matrix (lower is better)")
plot_matrix(cost_matrix, title="Cost Reduction Matrix (higher is better)")
