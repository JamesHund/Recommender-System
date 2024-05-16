from typing import Tuple, List
import csv
import numpy as np
from tqdm import tqdm
import time
import random

class Movie:

    def __init__(self, movie_id, ratings, ratings_test, title = "", genres = []):
        self.movie_id = movie_id
        self.title = title
        self.genres = genres
        self.ratings = ratings
        self.ratings_test = ratings_test

class User:

    def __init__(self, uid, ratings, ratings_test):
        self.uid = uid
        self.ratings = ratings
        self.ratings_test = ratings_test

class Recommender:

    def __init__(self, k = 20, gamma=0.02, lam=0.05, tau=0.05):
        #map user id to to user index
        self.uid_map = {}
        #map movie id to movie index
        self.mid_map = {}
        #map movie title to movie id
        self.movie_title_to_id = {}

        self.users = []
        self.movies = []
        self.k = k

        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        print("Initialized")
    
    def set_k(self, k):
        self.k = k
    
    def set_params(self, gamma, lam, tau):
        self.gamma = gamma
        self.lam = lam
        self.tau = tau

    #populate users and movies adjacency list
    def initialize_from_csv(self, movies_csv_location, ratings_csv_location, train_test_split=False, train_ratio = 0.8):

        self.train_test_split = train_test_split

        #ratings
        with open(ratings_csv_location, mode='r', encoding='utf-8') as file:
            # Create a CSV reader object
            reader = csv.DictReader(file)

            # Iterate over the rows
            u_index = 0
            m_index = 0
            self.max_uid = -1
            for row in reader:
                user_id = int(row['userId'])
                movie_id = int(row['movieId'])
                rating = float(row['rating'])

                if user_id > self.max_uid:
                    self.max_uid = user_id

                is_training_instance = True

                if train_test_split:
                    is_training_instance = random.random() < train_ratio

                if movie_id not in self.mid_map:
                    self.mid_map[movie_id] = m_index
                    self.movies.append(Movie(movie_id, [], []))
                    m_index += 1

                user_rating_tuple = (self.mid_map[movie_id], rating)

                #check if user has been created for user_id
                if user_id not in self.uid_map:
                    self.uid_map[user_id] = u_index

                    if is_training_instance:
                        self.users.append(User(user_id, [user_rating_tuple], []))
                    else:
                        self.users.append(User(user_id, [], [user_rating_tuple]))
                    u_index += 1
                else:
                    if is_training_instance:
                        self.users[self.uid_map[user_id]].ratings.append(user_rating_tuple)
                    else:
                        self.users[self.uid_map[user_id]].ratings_test.append(user_rating_tuple)

                movie_rating_tuple = (self.uid_map[user_id], rating)

                if is_training_instance:
                    self.movies[self.mid_map[movie_id]].ratings.append(movie_rating_tuple)
                else:
                    self.movies[self.mid_map[movie_id]].ratings_test.append(movie_rating_tuple)
        
        print(f"Parsed {ratings_csv_location}")

        with open(movies_csv_location, mode='r', encoding='utf-8') as file:
            # Create a CSV reader object
            reader = csv.DictReader(file)

            # Iterate over the rows
            for row in reader:
                movie_id = int(row['movieId'])
                title = row['title']
                genres = row['genres'].split('|')

                if movie_id in self.mid_map:
                    self.movie_title_to_id[title] = movie_id
                    self.movies[self.mid_map[movie_id]].title = title
                    self.movies[self.mid_map[movie_id]].genres = genres
        
        print(f"Parsed {movies_csv_location}")
        
        for user in tqdm(self.users, "Reformatting user ratings"):
            user.ratings = np.array(user.ratings)

            if train_test_split:
                user.ratings_test = np.array(user.ratings_test)

        for movie in tqdm(self.movies, "Reformatting movie ratings"):
            movie.ratings = np.array(movie.ratings) 

            if train_test_split:
                movie.ratings_test = np.array(movie.ratings_test)
    
    def insert_dummy_user(self, ratings):
        uid = self.max_uid + 1

        self.uid_map[uid] = len(self.users)
        user = User(uid, [], [])
        self.users.append(user)
        #ratings is a list containing a list of tuples
        for movie_title, rating in ratings:
            mid = self.movie_title_to_id[movie_title]
            assert mid is not None
            movie_rating = np.array([self.uid_map[uid], rating])
            self.movies[self.mid_map[mid]].ratings = np.vstack((self.movies[self.mid_map[mid]].ratings, movie_rating))
            user.ratings.append((self.mid_map[mid], rating))

        user.ratings = np.array(user.ratings)

        self.max_uid += 1
        return uid

    def predict_movies_for_user(self, uid, biases_only=False):
        #return movie recommendations
        m = self.uid_map[uid]
        user_embedding = self.U[m, :]
        
        expected_rating = self.movie_biases + self.user_biases[m]

        if not biases_only:
            inner_products = self.V.dot(user_embedding)
            expected_rating = expected_rating + inner_products

        sorted_indices = np.argsort(-expected_rating)
        
        ratings = expected_rating[sorted_indices]
        titles = np.array([self.movies[idx].title for idx in sorted_indices])
        return ratings, titles
            
    def metrics(self, test=False):
        if not test:
            U = self.U
            V = self.V
            user_biases = self.user_biases
            movie_biases = self.movie_biases
        else:
            U = self.U_test
            V = self.V_test
            user_biases = self.user_biases_test
            movie_biases = self.movie_biases_test

        # Initialize sums and counts
        sum_ratings_nll = 0.0  # Sum of squared errors for neg_log_likelihood
        sum_rankings_rmse = 0.0  # Sum of squared errors for RMSE
        num_ratings = 0  # Count of ratings for RMSE

        # Calculate errors for each user's ratings
        for m, user in enumerate(self.users):
            user_vector = U[m, :]
            user_bias = user_biases[m]

            if not test:
                ratings = user.ratings
            else:
                ratings = user.ratings_test

            if len(ratings) > 0:
                for n, rating in ratings:
                    item_vector = V[int(n), :]
                    item_bias = movie_biases[int(n)]

                    # Compute prediction error
                    prediction_error = rating - (user_vector.dot(item_vector) + user_bias + item_bias)
                    
                    # Update sums for neg_log_likelihood
                    sum_ratings_nll += prediction_error**2
                    
                    # Update sums for RMSE
                    sum_rankings_rmse += prediction_error**2
                    num_ratings += 1

        # Sum of squares of user feature vectors and movie feature vectors
        user_sum = np.sum(U**2) * self.tau / 2
        movie_sum = np.sum(V**2) * self.tau / 2

        # Bias regularization terms
        bias_user_sum = np.sum(user_biases**2)
        bias_movie_sum = np.sum(movie_biases**2)

        # Combine all terms for neg_log_likelihood
        neg_log_likelihood = (self.lam / 2) * sum_ratings_nll + (self.tau / 2) * (bias_user_sum + bias_movie_sum) + user_sum + movie_sum

        # Compute RMSE
        RMSE = np.sqrt(sum_rankings_rmse / num_ratings)

        return neg_log_likelihood, RMSE

    def metrics_biases_only(self, test=False):
        if not test:
            user_biases = self.user_biases
            movie_biases = self.movie_biases
        else:
            user_biases = self.user_biases_test
            movie_biases = self.movie_biases_test

        # Initialize sums and counts
        sum_ratings_nll = 0.0  # Sum of squared errors for neg_log_likelihood
        sum_rankings_rmse = 0.0  # Sum of squared errors for RMSE
        num_ratings = 0  # Count of ratings for RMSE

        # Calculate errors for each user's ratings
        for m, user in enumerate(self.users):
            user_bias = user_biases[m]

            if not test:
                ratings = user.ratings
            else:
                ratings = user.ratings_test
            for n, rating in ratings:
                item_bias = movie_biases[int(n)]

                # Compute prediction error
                prediction_error = rating - (user_bias + item_bias)
                
                # Update sums for neg_log_likelihood
                sum_ratings_nll += prediction_error**2
                
                # Update sums for RMSE
                sum_rankings_rmse += prediction_error**2
                num_ratings += 1

        # Bias regularization terms
        bias_user_sum = np.sum(user_biases**2)
        bias_movie_sum = np.sum(movie_biases**2)

        # Combine all terms for neg_log_likelihood
        neg_log_likelihood = (self.lam / 2) * sum_ratings_nll + (self.tau / 2) * (bias_user_sum + bias_movie_sum)

        # Compute RMSE
        RMSE = np.sqrt(sum_rankings_rmse / num_ratings)

        return neg_log_likelihood, RMSE

    
    def init_statistics(self, biases_only=False, extra_stats=False):
        suffixes = [""]
        if self.train_test_split:
            suffixes = ["", "_test"]
        
        statistics = {} 
        for suffix in suffixes:
            statistics[f'neg_log_liks{suffix}'] = []
            statistics[f'RMSEs{suffix}'] = []

            if extra_stats:
                statistics[f'mean_user_bias{suffix}'] = []
                statistics[f'mean_item_bias{suffix}'] = []

                if not biases_only:
                    statistics[f'user_embed_length{suffix}'] = []
                    statistics[f'item_embed_length{suffix}'] = []
        
        return statistics

    def update_statistics(self, statistics, biases_only=False, extra_stats=False):
        suffixes = [""]
        if self.train_test_split:
            suffixes = ["", "_test"]
        
        for suffix in suffixes:
            test = "test" in suffix
            if extra_stats:
                statistics[f'mean_user_bias{suffix}'].append(np.mean(self.user_biases))
                statistics[f'mean_item_bias{suffix}'].append(np.mean(self.movie_biases))

            if not biases_only:
                neg_log_lik, RMSE = self.metrics(test=test)

                statistics[f'neg_log_liks{suffix}'].append(neg_log_lik)
                statistics[f'RMSEs{suffix}'].append(RMSE)

                if extra_stats:
                    statistics[f'user_embed_length{suffix}'].append(Recommender.avg_inner_product(self.U))
                    statistics[f'item_embed_length{suffix}'].append(Recommender.avg_inner_product(self.V))
            else:
                neg_log_lik, RMSE = self.metrics_biases_only(test=test)

                statistics[f'neg_log_liks{suffix}'].append(neg_log_lik)
                statistics[f'RMSEs{suffix}'].append(RMSE)

    def update_user_bias(lam, gamma, ratings, user_embedding, item_embeddings, item_biases):
        if len(ratings) == 0:
            return 0.0

        indices = ratings[:, 0].astype(int)
        actual_ratings = ratings[:, 1]

        # Compute the dot product between the user_embedding and the corresponding item_embeddings
        item_embedding_contrib = item_embeddings[indices].dot(user_embedding)

        # Compute contributions from item_biases
        item_bias_contrib = item_biases[indices]

        # Sum all contributions
        summ = np.sum(actual_ratings - item_embedding_contrib - item_bias_contrib)

        # Calculate coefficient
        coeff = lam / (gamma + lam * len(actual_ratings))

        return coeff * summ

    def update_item_bias(lam, gamma, ratings, item_embedding, user_embeddings, user_biases):
        if len(ratings) == 0:
            return 0.0

        indices = ratings[:, 0].astype(int)
        actual_ratings = ratings[:, 1]

        # Compute the dot product between the item_embedding and the corresponding user_embeddings
        user_embedding_contrib = user_embeddings[indices].dot(item_embedding)

        # Compute contributions from user_biases
        user_bias_contrib = user_biases[indices]

        # Sum all contributions
        summ = np.sum(actual_ratings - user_embedding_contrib - user_bias_contrib)

        # Calculate coefficient
        coeff = lam / (gamma + lam * len(actual_ratings))

        return coeff * summ

    def update_user_embedding(lam, tau, V, ratings, user_bias, item_biases):
        N, k = V.shape

        if len(ratings) == 0:
            return np.zeros((k, ))

        item_indices = ratings[:, 0].astype(int)
        r_vec = ratings[:, 1].reshape((-1, 1))
        item_biases_vec = item_biases[item_indices].reshape((-1, 1))

        vec = r_vec - item_biases_vec - user_bias

        # sum should be a k dimensional vector
        num_sum = lam * (V[item_indices, :].T @ vec)

        # Compute the sum of outer products using einsum
        outer_matrix = np.einsum('bi,bj->ij', V[item_indices, :], V[item_indices, :])

        matrix = lam * outer_matrix + tau * np.eye(k)

        return np.linalg.solve(matrix, num_sum).flatten()

    #update_user_embedding is the exact same
    def update_item_embedding(lam, tau, U, ratings, item_bias, user_biases):
        return Recommender.update_user_embedding(lam, tau, U, ratings, item_bias, user_biases)

    def avg_inner_product(matrix):
        inner_products = np.einsum('ij,ij->i', matrix, matrix)
        return np.mean(inner_products)

    def fit(self, max_iter=20, extra_stats=False):
        start_time = time.time()
        M = len(self.users)
        N = len(self.movies)

        #user biases
        self.user_biases = np.zeros((M))
        #movie biases
        self.movie_biases = np.zeros((N))
        #initialize U and V matrices
        self.U = np.random.normal(0, 1/np.sqrt(self.k), size=(M, self.k))
        self.V = np.random.normal(0, 1/np.sqrt(self.k), size=(N, self.k))

        if self.train_test_split:
            #user biases
            self.user_biases_test = np.zeros((M))
            #movie biases
            self.movie_biases_test = np.zeros((N))
            #initialize U and V matrices
            self.U_test = np.random.normal(0, 1/np.sqrt(self.k), size=(M, self.k))
            self.V_test = np.random.normal(0, 1/np.sqrt(self.k), size=(N, self.k))


        statistics = self.init_statistics(extra_stats=extra_stats)
        #self.update_statistics(statistics, extra_stats=extra_stats)

        elapsed_time = time.time() - start_time
        print(f"Initialized variables and statistics: {elapsed_time}s")


        #initial update to user and item biases
        #update user parameters
        start_time = time.time()
        embedding_zeros = np.zeros((self.k, ))
        U_zeros = np.zeros_like(self.U)
        V_zeros = np.zeros_like(self.V)
        for m, user in zip(range(M), self.users):
            ratings = user.ratings
            #---user biases---
            self.user_biases[m] = Recommender.update_user_bias(self.lam, self.gamma, ratings, embedding_zeros, V_zeros, self.movie_biases)

            if self.train_test_split:
                ratings_test = user.ratings_test
                self.user_biases_test[m] = Recommender.update_user_bias(self.lam, self.gamma, ratings_test, embedding_zeros, V_zeros, self.movie_biases_test)

        for n, item in zip(range(N), self.movies):
            ratings = item.ratings
            #---item biases---
            self.movie_biases[n] = Recommender.update_item_bias(self.lam, self.gamma, ratings, embedding_zeros, U_zeros, self.user_biases)

            if self.train_test_split:
                ratings_test = item.ratings_test
                #---item biases---
                self.movie_biases_test[n] = Recommender.update_item_bias(self.lam, self.gamma, ratings, embedding_zeros, U_zeros, self.user_biases)
        
        self.update_statistics(statistics, extra_stats=extra_stats)
        elapsed_time = time.time() - start_time
        print(f"Ran initial update to user and item biases: {elapsed_time}s")

        for it in tqdm(range(max_iter)):
            #update user parameters
            for m, user in zip(range(M), self.users):
                ratings = user.ratings
                user_embedding = self.U[m,:]
                #---user biases---
                self.user_biases[m] = Recommender.update_user_bias(self.lam, self.gamma, ratings, user_embedding, self.V, self.movie_biases)
                #---user vectors---
                user_embedding = Recommender.update_user_embedding(self.lam, self.tau, self.V, ratings, self.user_biases[m], self.movie_biases)
                self.U[m,:] = user_embedding

                if self.train_test_split:
                    ratings_test = user.ratings_test
                    user_embedding_test = self.U_test[m,:]
                    #---user biases---
                    self.user_biases_test[m] = Recommender.update_user_bias(self.lam, self.gamma, ratings_test, user_embedding_test, self.V_test, self.movie_biases_test)
                    #---user vectors---
                    user_embedding_test = Recommender.update_user_embedding(self.lam, self.tau, self.V_test, ratings_test, self.user_biases_test[m], self.movie_biases_test)
                    self.U_test[m,:] = user_embedding_test
            
            #update movie parameters
            for n, item in zip(range(N), self.movies):
                ratings = item.ratings
                item_embedding = self.V[n,:]
                #---item biases---
                self.movie_biases[n] = Recommender.update_item_bias(self.lam, self.gamma, ratings, item_embedding, self.U, self.user_biases)
                #---item vectors---
                item_embedding = Recommender.update_item_embedding(self.lam, self.tau, self.U, ratings, self.movie_biases[n], self.user_biases)
                self.V[n, :] = item_embedding
            
                if self.train_test_split:
                    ratings_test = item.ratings_test
                    item_embedding_test = self.V_test[n,:]
                    #---item biases---
                    self.movie_biases_test[n] = Recommender.update_item_bias(self.lam, self.gamma, ratings_test, item_embedding_test, self.U_test, self.user_biases_test)
                    #---item vectors---
                    item_embedding_test = Recommender.update_item_embedding(self.lam, self.tau, self.U_test, ratings_test, self.movie_biases_test[n], self.user_biases_test)
                    self.V_test[n, :] = item_embedding_test

            self.update_statistics(statistics, extra_stats=extra_stats)

        return statistics 

    def fit_biases_only(self, max_iter=20, extra_stats=False):
        M = len(self.users)
        N = len(self.movies)

        #user biases
        self.user_biases = np.zeros((M))
        #movie biases
        self.movie_biases = np.zeros((N))

        if self.train_test_split:
            self.user_biases_test = np.zeros((M))
            self.movie_biases_test = np.zeros((N))

        #initialize U and V matrices
        self.U = np.zeros((M, self.k)) 
        self.V = np.zeros((N, self.k)) 
        embedding_zeros = np.zeros((self.k, ))

        statistics = self.init_statistics(biases_only=True, extra_stats=extra_stats)
        #self.update_statistics(statistics, biases_only=True, extra_stats=extra_stats)

        for it in tqdm(range(max_iter)):
            #update user parameters
            for m, user in zip(range(M), self.users):
                ratings = user.ratings
                #---user biases---
                self.user_biases[m] = Recommender.update_user_bias(self.lam, self.gamma, ratings, embedding_zeros, self.V, self.movie_biases)

                if self.train_test_split:
                    ratings_test = user.ratings_test
                    #---user biases---
                    self.user_biases_test[m] = Recommender.update_user_bias(self.lam, self.gamma, ratings_test, embedding_zeros, self.V, self.movie_biases_test)


            #update movie parameters
            for n, item in zip(range(N), self.movies):
                ratings = item.ratings
                #---item biases---
                self.movie_biases[n] = Recommender.update_item_bias(self.lam, self.gamma, ratings, embedding_zeros, self.U, self.user_biases)
                if self.train_test_split:
                    ratings_test = item.ratings_test
                    #---item biases---
                    self.movie_biases_test[n] = Recommender.update_item_bias(self.lam, self.gamma, ratings_test, embedding_zeros, self.U, self.user_biases_test)

            self.update_statistics(statistics, biases_only=True, extra_stats=extra_stats)

        return statistics 

def run_cmdline():
    rec = Recommender(lam=0.01, gamma=0.01, tau=0.01)
    rec.initialize_from_csv("ml-latest-small/movies.csv", "ml-latest-small/ratings.csv")
    print(len(rec.users))
    print(len(rec.movies))