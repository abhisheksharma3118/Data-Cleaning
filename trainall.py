import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class MachineLearning:
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        
    def train_model(self, X_train, y_train):
        if self.model_name == 'Logistic Regression':
            self.model = LogisticRegression()
        elif self.model_name == 'Decision Tree':
            self.model = DecisionTreeClassifier()
        elif self.model_name == 'Random Forest':
            self.model = RandomForestClassifier()
        else:
            raise ValueError('Invalid model name')
        
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        if self.model is None:
            raise ValueError('Model not trained')
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError('Model not trained')
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    
if __name__ == '__main__':
    # Load data
    data = pd.read_csv('titanic.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate Logistic Regression model
    lr = MachineLearning('Logistic Regression')
    lr.train_model(X_train, y_train)
    print('Logistic Regression accuracy:', lr.evaluate(X_test, y_test))
    
    # Train and evaluate Decision Tree model
    dt = MachineLearning('Decision Tree')
    dt.train_model(X_train, y_train)
    print('Decision Tree accuracy:', dt.evaluate(X_test, y_test))
    
    # Train and evaluate Random Forest model
    rf = MachineLearning('Random Forest')
    rf.train_model(X_train, y_train)
    print('Random Forest accuracy:', rf.evaluate(X_test, y_test))
