#!pip install imblearn

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.under_sampling import RandomUnderSampler

class ResampledEnsemble(BaseEstimator):
  def __init__(self, base_estimator=DecisionTree(),n_estimators=100, max_depth=None, max_features=None,min_samples_split=2,min_samples_leaf=1):
   
    self._estimator_type='classifier'
    self.base_estimator=base_estimator
    self.n_estimators=n_estimators
    self.max_depth=max_depth
    self.max_features=max_features
    self.min_samples_split=min_samples_split
    self.min_samples_leaf= min_samples_leaf
    
    self.estimators=self._generate_estimators()
    self.estimator=VotingClassifier(self.estimators,voting="Soft")
    

  def _generate_estimators():
    estimators=[]
    for i in range(self.n_estimators):
      est=clone(self.base_estimator)
      est.random_state=i
      
      est.max_depth=self.max_depth
      est.max_features=self.max_features
      est.min_samples_split=self.min_samples_split
      est.min_samples_leaf=self.min_samples_leaf
      
      
      pipe=make_imb_pipeline(RandomUnderSampler(random_state=i,replacement=True),est)
      estimators.append((f"est{i}", pipe))
      
    return estimators
  def fit(self,x,y,sample_weight=None):
    return self.estimator.fit(X,y,sample_weight)
  
  def predict(self,x):
    return self.estimator.predict(x)
    
    
  def classes_(self):
    if self.estimator:
      return self.estimator.classes_
      
      
      
  def set_params(self,**params):
    if not params:
      return self
    for key, value in params.items():
      if hasattr(self,key):
        setattr(self,key,value)
      else:
        self.kwargs[key]=value
    self.estimators=self._generate_estimators()
    self.estimator=VotingClassifier(self.estimators,voting="soft")
    return self
