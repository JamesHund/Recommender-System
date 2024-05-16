import matplotlib.pylab as plt
import math
import numpy as np
import seaborn as sns

def plot_user_movie_ratings(rec):
    user_freq = {}
    movie_freq = {}

    #iterate over users
    for user in rec.users:
        degree = len(user.ratings)
        if degree not in user_freq:
            user_freq[degree] = 1
        else:
            user_freq[degree] += 1

    for movie in rec.movies:
        degree = len(movie.ratings)

        if degree not in movie_freq:
            movie_freq[degree] = 1
        else:
            movie_freq[degree] += 1

    m_deg = [degree for degree in movie_freq.keys()]
    m_freq = [freq for freq in movie_freq.values()]
    plt.scatter(m_deg, m_freq, c='red', s=2, label='Movies')

    u_deg = [degree for degree in user_freq.keys()]
    u_freq = [freq for freq in user_freq.values()]
    plt.scatter(u_deg, u_freq, c='blue', s=2, label='Users')
    plt.xscale("log")
    plt.yscale("log")


    plt.xlabel("Log Degree")
    plt.ylabel("Log Frequency")
    plt.legend()
    plt.title("Users and Movies Rating Distribution")
    plt.savefig('plots/user_movie_ratings.pdf', format='pdf', transparent=True)
    plt.show()

def plot_ratings_distribution(rec):
    # Initialize a dictionary to store frequency of each rating value
    rating_freq = {}
    
    # Iterate over all ratings in the dataset
    for user in rec.users:
        for rating in user.ratings:
            # Assume rating is a tuple (movie_id, rating_value)
            rating_value = rating[1]
            if rating_value not in rating_freq:
                rating_freq[rating_value] = 1
            else:
                rating_freq[rating_value] += 1

    # Prepare data for plotting
    ratings = sorted(rating_freq.keys())
    frequencies = [rating_freq[r] for r in ratings]

    # Adjust bar width for finer increments in ratings
    bar_width = 0.4 if 0.5 in ratings else 1

    # Create a bar chart to visualize the frequency of each rating value
    plt.bar(ratings, frequencies, width=bar_width, color='green')
    plt.xticks(ratings)

    # Set the labels and title
    plt.xlabel("Rating Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Ratings")
    
    # Show the plot
    plt.savefig('plots/ratings_dist.pdf', format='pdf', transparent=True)
    plt.show()

def plot_ratings_distribution_kde(rec):
    # Gather all rating values into a list
    ratings_list = []

    # Iterate over all ratings in the dataset
    for user in rec.users:
        for rating in user.ratings:
            ratings_list.append(rating[1])

    # Convert list of ratings to a numpy array
    ratings_array = np.array(ratings_list)

    sns.kdeplot(ratings_array, bw_adjust=8, fill=True)

    # Setting labels and title
    plt.xlabel("Rating Value")
    plt.ylabel("Density")
    plt.title("Estimated Distribution of Ratings")
    plt.savefig('plots/ratings_dist_kde.pdf', format='pdf', transparent=True)
    plt.show()

def plot_statistics(statistics, train_test_split=True, biases_only=False, extra_stats=False, save_suffix=""):
    # Create two figures: one for biases and embeddings, another for RMSE and NLL
    fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))   # RMSE and NLL
    plt.tight_layout(pad=4.0)

    # Plot RMSE
    axes2[0].plot(statistics['RMSEs'], label='Train RMSE')
    if train_test_split:
        axes2[0].plot(statistics['RMSEs_test'], label='Test RMSE', linestyle='--')
    axes2[0].set_title('Root Mean Square Error')
    axes2[0].set_xlabel('Epoch')
    axes2[0].set_ylabel('RMSE')
    axes2[0].legend()

    # Plot Negative Log Likelihood
    axes2[1].plot(statistics['neg_log_liks'], label='Train NLL')
    if train_test_split:
        axes2[1].plot(statistics['neg_log_liks_test'], label='Test NLL', linestyle='--')
    axes2[1].set_title('Negative Log Likelihood')
    axes2[1].set_xlabel('Epoch')
    axes2[1].set_ylabel('NLL')
    axes2[1].legend()

    plt.savefig(f"plots/metrics_{save_suffix}.pdf", format='pdf', transparent=True)

    if extra_stats:
        fig1, axes1 = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))  # Biases and Embeddings
        # Plot Mean User Bias
        axes1[0, 0].plot(statistics['mean_user_bias'], label='Mean User Bias')
        if train_test_split:
            axes1[0, 0].plot(statistics['mean_user_bias_test'], label='Mean User Bias Test', linestyle='--')
        axes1[0, 0].set_title('Mean User Bias')
        axes1[0, 0].set_xlabel('Epoch')
        axes1[0, 0].set_ylabel('Bias')
        axes1[0, 0].legend()

        # Plot Mean Item Bias
        axes1[0, 1].plot(statistics['mean_item_bias'], label='Mean Item Bias')
        if train_test_split:
            axes1[0, 1].plot(statistics['mean_item_bias_test'], label='Mean Item Bias Test', linestyle='--')
        axes1[0, 1].set_title('Mean Item Bias')
        axes1[0, 1].set_xlabel('Epoch')
        axes1[0, 1].set_ylabel('Bias')
        axes1[0, 1].legend()

        # Only plot the embedding lengths if not a biases_only model
        if not biases_only:
            axes1[1, 0].plot(statistics['user_embed_length'], label='User Embedding Length')
            if train_test_split:
                axes1[1, 0].plot(statistics['user_embed_length_test'], label='User Embedding Length Test', linestyle='--')
            axes1[1, 0].set_title('User Embedding Length')
            axes1[1, 0].set_xlabel('Epoch')
            axes1[1, 0].set_ylabel('Length')
            axes1[1, 0].legend()

            axes1[1, 1].plot(statistics['item_embed_length'], label='Item Embedding Length')
            if train_test_split:
                axes1[1, 1].plot(statistics['item_embed_length_test'], label='Item Embedding Length Test', linestyle='--')
            axes1[1, 1].set_title('Item Embedding Length')
            axes1[1, 1].set_xlabel('Epoch')
            axes1[1, 1].set_ylabel('Length')
            axes1[1, 1].legend()
        
        plt.savefig(f"plots/extra_stats_{save_suffix}.pdf", format='pdf', transparent=True)

    plt.show()