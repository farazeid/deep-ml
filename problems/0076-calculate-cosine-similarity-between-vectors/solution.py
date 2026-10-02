import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	# Implement your code here
	if v1.shape != v2.shape:
		raise ValueError(f"v1 and v2 must have the same shape, not {v1.shape} and {v2.shape} respectively.")
	if len(v1) == 0:
		raise ValueError(f"v1 {v1} cannot be empty.")
	if len(v2) == 0:
		raise ValueError(f"v2 {v2} cannot be empty.")
	v1_norm = np.linalg.norm(v1)
	v2_norm = np.linalg.norm(v2)
	if v1_norm == 0:
		raise ValueError(f"v1's L2 norm {v1_norm} cannot be zero.")
	if v2_norm == 0:
		raise ValueError(f"v2's L2 norm {v2_norm} cannot be zero.")
	return np.dot(v1, v2) / (v1_norm * v2_norm)