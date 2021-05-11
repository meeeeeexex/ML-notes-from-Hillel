import matplotlib.pyplot as plt

def plot_model(model,qualifies_df):
    plt.xlabel('Technical grade')
    plt.ylabel('English grade')

    qualified_candidates = qualifies_df[qualifies_df['qualifies']==1]
    unqualified_candidates = qualifies_df[qualifies_df['qualifies']==0]

    max_grade = 101
    pred_points = []

    for eng_grade in range(max_grade):
        for tech_grade in range(max_grade):
            pred_points.append([tech_grade,eng_grade])

    probability_levels = model.predict_proba(pred_points)[:,1]
    probability_matrix = probability_levels.reshape(max_grade,max_grade)

    plt.contourf(probability_matrix, cmap= 'rainbow')#cmap = 'RdYlBu'/'binary'
    plt.scatter(qualified_candidates['technical_grade'],qualified_candidates['english_grade'],color='w')
    plt.scatter(unqualified_candidates['technical_grade'],unqualified_candidates['english_grade'],color='k')