from dataclasses import dataclass
from typing import Tuple, List
import csv
import numpy as np
from tqdm import tqdm
from numba import njit
import time

class Movie:

    def __init__(self, movie_id, ratings, title = "", genres = []):
        self.movie_id = movie_id
        self.title = title
        self.genres = genres
        self.ratings = ratings

class User:

    def __init__(self, uid, ratings):
        self.uid = uid
        self.ratings = ratings


class Recommender:

    def __init__(self, k = 20, gamma=0.02, lam=0.05, tau=0.05):
        self.uid_map = {}
        self.mid_map = {}
        self.movie_title_to_id = {}

        self.users = []
        self.movies = []
        self.k = k

        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        print("Initialized")

    #populate users and movies adjacency list
    def initialize_from_csv(self, movies_csv_location, ratings_csv_location):
        #ratings
        with open(ratings_csv_location, mode='r', encoding='utf-8') as file:
            # Create a CSV reader object
            reader = csv.DictReader(file)

            # Iterate over the rows
            u_index = 0
            m_index = 0
            for row in reader:
                user_id = int(row['userId'])
                movie_id = int(row['movieId'])
                rating = float(row['rating'])

                if movie_id not in self.mid_map:
                    self.mid_map[movie_id] = m_index
                    self.movies.append(Movie(movie_id, []))
                    m_index += 1

                user_rating_tuple = (self.mid_map[movie_id], rating)

                #check if user has been created for user_id
                if user_id not in self.uid_map:
                    self.uid_map[user_id] = u_index
                    self.users.append(User(user_id, [user_rating_tuple]))
                    u_index += 1
                else:
                    self.users[self.uid_map[user_id]].ratings.append(user_rating_tuple)

                movie_rating_tuple = (self.uid_map[user_id], rating)

                self.movies[self.mid_map[movie_id]].ratings.append(movie_rating_tuple)
        
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

        for movie in tqdm(self.movies, "Reformatting movie ratings"):
            assert len(movie.ratings) != 0
            movie.ratings = np.array(movie.ratings) 
            
    def neg_log_likelihood(self):
        sum_ratings = 0.0
        user_sum = 0.0

        for m, user in zip(range(len(self.users)), self.users):
            for n, rank in user.ratings:
                sum_ratings += (rank - self.U[m,:].dot(self.V[int(n),:]) - self.user_biases[m] - self.movie_biases[int(n)])**2
            
            user_sum += self.U[m,:].dot(self.U[m,:])
        
        user_sum *= self.tau/2

        movie_sum = 0.0
        for n in range(len(self.movies)):
            movie_sum += self.V[n,:].dot(self.V[n,:])
        
        movie_sum *= self.tau/2

        return (self.lam/2)*sum_ratings + (self.tau/2)*(self.user_biases.dot(self.user_biases) + self.movie_biases.dot(self.movie_biases)) + user_sum + movie_sum

    def RMSE(self):
        sum_rankings = 0.0

        num_ratings = 0
        for m, user in zip(range(len(self.users)), self.users):
            for n, rank in user.ratings:
                sum_rankings += (rank - self.U[m,:].dot(self.V[int(n),:]) - self.user_biases[m] - self.movie_biases[int(n)])**2
                num_ratings += 1
        
        
        return np.sqrt(sum_rankings/num_ratings)
    

    def combined_metrics_efficient(self):
        # Initialize sums and counts
        sum_ratings_nll = 0.0  # Sum of squared errors for neg_log_likelihood
        sum_rankings_rmse = 0.0  # Sum of squared errors for RMSE
        num_ratings = 0  # Count of ratings for RMSE

        # Calculate errors for each user's ratings
        for m, user in enumerate(self.users):
            user_vector = self.U[m, :]
            user_bias = self.user_biases[m]

            for n, rank in user.ratings:
                item_vector = self.V[int(n), :]
                item_bias = self.movie_biases[int(n)]

                # Compute prediction error
                prediction_error = rank - (user_vector.dot(item_vector) + user_bias + item_bias)
                
                # Update sums for neg_log_likelihood
                sum_ratings_nll += prediction_error**2
                
                # Update sums for RMSE
                sum_rankings_rmse += prediction_error**2
                num_ratings += 1

        # Sum of squares of user feature vectors and movie feature vectors
        user_sum = np.sum(self.U**2) * self.tau / 2
        movie_sum = np.sum(self.V**2) * self.tau / 2

        # Bias regularization terms
        bias_user_sum = np.sum(self.user_biases**2)
        bias_movie_sum = np.sum(self.movie_biases**2)

        # Combine all terms for neg_log_likelihood
        neg_log_likelihood = (self.lam / 2) * sum_ratings_nll + (self.tau / 2) * (bias_user_sum + bias_movie_sum) + user_sum + movie_sum

        # Compute RMSE
        RMSE = np.sqrt(sum_rankings_rmse / num_ratings)

        return neg_log_likelihood, RMSE

    def update_user_bias(lam, gamma, ratings, user_embedding, item_embeddings, item_biases):
        coeff = lam/(gamma + lam*ratings.shape[0])

        summ = 0.0
        for n, rating in ratings:
            summ += rating
            summ -= user_embedding.dot(item_embeddings[int(n)])
            summ -= item_biases[int(n)]
        
        return coeff*summ
    
    def update_user_bias_vectorized(lam, gamma, ratings, user_embedding, item_embeddings, item_biases):
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
        coeff = lam/(gamma + lam*ratings.shape[0])

        summ = 0.0
        for m, rating in ratings:
            summ += rating
            summ -= item_embedding.dot(user_embeddings[int(m)])
            summ -= user_biases[int(m)]
        
        return coeff*summ

    def update_item_bias_vectorized(lam, gamma, ratings, item_embedding, user_embeddings, user_biases):
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

        item_indices = ratings[:,0].astype(int)
        r_vec = ratings[:,1].reshape((-1, 1))
        item_biases_vec = item_biases[item_indices].reshape((-1, 1))

        vec = r_vec - item_biases_vec - user_bias

        #sum should be be a k dimensional vector
        num_sum = lam*(V[item_indices, :].T @ vec)

        outer_matrix = np.zeros((k, k))
        for row in V[item_indices, :]:
            outer_matrix += np.outer(row, row)

        matrix = lam*outer_matrix + tau*np.eye(k)

        return np.linalg.solve(matrix, num_sum.reshape((-1,1))).reshape((k,))
    
    def update_user_embedding_optimized(lam, tau, V, ratings, user_bias, item_biases):
        N, k = V.shape

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

    def update_item_embedding(lam, tau, user_embeddings, ratings, item_bias, user_biases):
        pass

    def avg_inner_product(matrix):
        inner_products = np.einsum('ij,ij->i', matrix, matrix)
        return np.mean(inner_products)
    
    def init_statistics():
        statistics = {} 
        statistics['neg_log_liks'] = []
        statistics['RMSEs'] = []
        statistics['user_embed_length'] = []
        statistics['item_embed_length'] = []
        statistics['mean_user_bias'] = []
        statistics['mean_item_bias'] = []
        return statistics

    def update_statistics(self, statistics):
        neg_log_lik, RMSE = self.combined_metrics_efficient()

        statistics['neg_log_liks'].append(neg_log_lik)
        statistics['RMSEs'].append(RMSE)
        statistics['user_embed_length'].append(Recommender.avg_inner_product(self.U))
        statistics['item_embed_length'].append(Recommender.avg_inner_product(self.V))
        statistics['mean_user_bias'].append(np.mean(self.user_biases))
        statistics['mean_item_bias'].append(np.mean(self.movie_biases))


    def fit_vectorized(self, max_iter=20):
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

        statistics = Recommender.init_statistics()
        self.update_statistics(statistics)

        elapsed_time = time.time() - start_time
        print(f"Initialized variables and calculated statistics: {elapsed_time}s")


        #initial update to user and item biases
        #update user parameters
        start_time = time.time()
        embedding_zeros = np.zeros((self.k, ))
        U_zeros = np.zeros_like(self.U)
        V_zeros = np.zeros_like(self.V)
        for m, user in zip(range(M), self.users):
            ratings = user.ratings
            #---user biases---
            self.user_biases[m] = Recommender.update_user_bias_vectorized(self.lam, self.gamma, ratings, embedding_zeros, V_zeros, self.movie_biases)

        for n, item in zip(range(N), self.movies):
            ratings = item.ratings
            #---item biases---
            self.movie_biases[n] = Recommender.update_item_bias_vectorized(self.lam, self.gamma, ratings, embedding_zeros, U_zeros, self.user_biases)
        
        elapsed_time = time.time() - start_time
        print(f"Ran initial update to user and item biases: {elapsed_time}s")

        start_time = time.time()
        self.update_statistics(statistics)
        elapsed_time_1 = time.time() - start_time
        print(f"Updated statistics: {elapsed_time_1}")

        for it in tqdm(range(max_iter)):
            #update user parameters
            start_time = time.time()
            for m, user in zip(range(M), self.users):
                ratings = user.ratings
                user_embedding = self.U[m,:]
                #---user biases---
                self.user_biases[m] = Recommender.update_user_bias_vectorized(self.lam, self.gamma, ratings, user_embedding, self.V, self.movie_biases)
                #---user vectors---
                user_embedding = Recommender.update_user_embedding_optimized(self.lam, self.tau, self.V, ratings, self.user_biases[m], self.movie_biases)
                self.U[m,:] = user_embedding
            
            elapsed_time_1 = time.time() - start_time
            print(f"Updated user embeddings and biases: {elapsed_time_1}")

            #update movie parameters
            start_time = time.time()
            for n, item in zip(range(N), self.movies):
                ratings = item.ratings
                item_embedding = self.V[n,:]
                #---item biases---
                self.movie_biases[n] = Recommender.update_item_bias_vectorized(self.lam, self.gamma, ratings, item_embedding, self.U, self.user_biases)
                #---item vectors---
                item_embedding = Recommender.update_user_embedding_optimized(self.lam, self.tau, self.U, ratings, self.movie_biases[n], self.user_biases)
                self.V[n, :] = item_embedding

            elapsed_time_2 = time.time() - start_time
            print(f"Updated item embeddings and biases: {elapsed_time_2}")
            
            start_time = time.time()
            self.update_statistics(statistics)
            elapsed_time_1 = time.time() - start_time
            print(f"Updated statistics: {elapsed_time_1}")

        return statistics 

def run_cmdline():
    rec = Recommender(lam=0.01, gamma=0.01, tau=0.01)
    rec.initialize_from_csv("ml-latest-small/movies.csv", "ml-latest-small/ratings.csv")
    print(len(rec.users))
    print(len(rec.movies))