from evaluatemod import run
import matplotlib.pyplot as plt

scores = []

for i in range(1, 41):
    print(i)
    _, averages = run('embeddings.pkl', selected_feature_groups=[2, 3, 4, 5, 6],
                      knn_k=i, echo=False)
    score = 0
    for method in ['across_areas', 'within_areas', 'individual_languages']:
        score += averages[method]['total']['score']
    score /= 3
    scores.append(score)
fig = plt.figure()
ax = fig.add_subplot(111, title="K in KNN")
ax.plot(range(1, 41), scores)
ax.set_xlabel('k')
ax.set_ylabel('Score')
plt.show()