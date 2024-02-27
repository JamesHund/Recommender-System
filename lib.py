from dataclasses import dataclass
from typing import Tuple, List
import csv
import numpy as np
from tqdm import tqdm

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

    def update_user_bias(lam, gamma, ratings, user_embedding, item_embeddings, item_biases):
        coeff = lam/(gamma + lam*ratings.shape[0])

        summ = 0.0
        for n, rating in ratings:
            summ += rating
            summ -= user_embedding.dot(item_embeddings[int(n)])
            summ -= item_biases[int(n)]
        
        return coeff*summ

    def update_item_bias(lam, gamma, ratings, item_embedding, user_embeddings, user_biases):
        coeff = lam/(gamma + lam*ratings.shape[0])

        summ = 0.0
        for m, rating in ratings:
            summ += rating
            summ -= item_embedding.dot(user_embeddings[int(m)])
            summ -= user_biases[int(m)]
        
        return coeff*summ

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

    def update_item_embedding(lam, tau, user_embeddings, ratings, item_bias, user_biases):
        pass

    def avg_inner_product(matrix):
        inner_products = np.einsum('ij,ij->i', matrix, matrix)
        return np.mean(inner_products)

    def fit_vectorized(self, max_iter=20):
        M = len(self.users)
        N = len(self.movies)

        #user biases
        self.user_biases = np.zeros((M))
        #movie biases
        self.movie_biases = np.zeros((N))

        #initialize U and V matrices
        #self.U = np.zeros((M, self.k))
        #self.V = np.zeros((N, self.k))
        self.U = np.random.normal(0, 1/np.sqrt(self.k), size=(M, self.k))
        self.V = np.random.normal(0, 1/np.sqrt(self.k), size=(N, self.k))

        statistics = {} 
        statistics['neg_log_liks'] = []
        statistics['RMSEs'] = []
        statistics['user_embed_length'] = []
        statistics['item_embed_length'] = []
        statistics['mean_user_bias'] = []
        statistics['mean_item_bias'] = []

        statistics['neg_log_liks'].append(self.neg_log_likelihood())
        statistics['RMSEs'].append(self.RMSE())
        statistics['user_embed_length'].append(Recommender.avg_inner_product(self.U))
        statistics['item_embed_length'].append(Recommender.avg_inner_product(self.V))
        statistics['mean_user_bias'].append(np.mean(self.user_biases))
        statistics['mean_item_bias'].append(np.mean(self.movie_biases))

        #initial update to user and item biases
        #update user parameters
        for m, user in zip(range(M), self.users):
            ratings = user.ratings
            #---user biases---
            self.user_biases[m] = Recommender.update_user_bias(self.lam, self.gamma, ratings, np.zeros((self.k, )), np.zeros_like(self.V), self.movie_biases)

        for n, item in zip(range(N), self.movies):
            ratings = item.ratings
            #---item biases---
            self.movie_biases[n] = Recommender.update_item_bias(self.lam, self.gamma, ratings, np.zeros((self.k, )), np.zeros_like(self.U), self.user_biases)
        
        statistics['neg_log_liks'].append(self.neg_log_likelihood())
        statistics['RMSEs'].append(self.RMSE())
        statistics['user_embed_length'].append(Recommender.avg_inner_product(self.U))
        statistics['item_embed_length'].append(Recommender.avg_inner_product(self.V))
        statistics['mean_user_bias'].append(np.mean(self.user_biases))
        statistics['mean_item_bias'].append(np.mean(self.movie_biases))

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
            
            #update movie parameters
            for n, item in zip(range(N), self.movies):
                ratings = item.ratings
                item_embedding = self.V[n,:]
                #---item biases---
                self.movie_biases[n] = Recommender.update_item_bias(self.lam, self.gamma, ratings, item_embedding, self.U, self.user_biases)
                #---item vectors---
                item_embedding = Recommender.update_user_embedding(self.lam, self.tau, self.U, ratings, self.movie_biases[n], self.user_biases)
                self.V[n, :] = item_embedding
            
            statistics['neg_log_liks'].append(self.neg_log_likelihood())
            statistics['RMSEs'].append(self.RMSE())
            statistics['user_embed_length'].append(Recommender.avg_inner_product(self.U))
            statistics['item_embed_length'].append(Recommender.avg_inner_product(self.V))
            statistics['mean_user_bias'].append(np.mean(self.user_biases))
            statistics['mean_item_bias'].append(np.mean(self.movie_biases))

        return statistics 

def run_cmdline():
    rec = Recommender(lam=0.01, gamma=0.01, tau=0.01)
    rec.initialize_from_csv("ml-latest-small/movies.csv", "ml-latest-small/ratings.csv")
    print(len(rec.users))
    print(len(rec.movies))