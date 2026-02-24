Option B - Exercice - Élaborez un modèle de scoring
Exercice

 

Prêt à résoudre l’exercice ? 
Barre

 

Dans cet exercice, vous allez réaliser une mission. 

 

Comment allez-vous procéder ?
Barre

 

Cette mission suit un scénario de projet professionnel, mais elle ne sera pas l’objet d’une soutenance. 

Vous pouvez suivre les étapes pour vous aider à réaliser vos livrables.

 

 Avant de démarrer, nous vous conseillons de :

lire toute la mission et ses documents liés ;

prendre des notes sur ce que vous avez compris ;

consulter les étapes pour vous guider ; 

préparer une liste de questions pour votre première session de mentorat.

 

Prêt à mener la mission ?
Barre

 

Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

 

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)

Voici les données dont vous aurez besoin pour réaliser l’algorithme de classification. Pour plus de simplicité, vous pouvez les télécharger à cette adresse.

Vous aurez besoin de joindre les différentes tables entre elles.

 

Votre mission :

Construire et optimiser un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.

Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un soucis de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle.

Mettre en œuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à la pré-production du modèle.

Michaël, votre manager, vous incite à sélectionner un ou des kernels Kaggle pour vous faciliter l’analyse exploratoire, la préparation des données et le feature engineering nécessaires à l’élaboration du modèle de scoring. 

Si vous le faites, vous devez analyser ce ou ces kernels et le ou les adapter pour vous assurer qu’il(s) répond(ent) aux besoins de votre mission.

Par exemple vous pouvez vous inspirer des kernels suivants : 

Pour l’analyse exploratoire

Pour la préparation des données et le feature engineering

C’est optionnel, mais nous vous encourageons à le faire afin de vous permettre de vous focaliser sur l’élaboration du modèle, son optimisation et sa compréhension.

 

De : Michaël

À : moi

Objet : Besoins et conseils pour l’élaboration d’un outil de credit scoring 

Bonjour,

 

Afin de pouvoir faire évoluer régulièrement le modèle, je souhaite tester la mise en œuvre une démarche de type MLOps d’automatisation et d’industrialisation de la gestion du cycle de vie du modèle. 

 

Vous trouverez en pièce jointe la liste d’outils à utiliser pour créer une plateforme MLOps qui s’appuie sur des outils Open Source. 

 

Je souhaite que vous puissiez mettre en oeuvre au minimum les étapes orientées MLOps suivantes : 

Dans le notebook d’entraînement des modèles, générer à l’aide de MLFlow un tracking d'expérimentations.

Lancer l’interface web “UI MLFlow" d'affichage des résultats du tracking.

Réaliser avec MLFlow un stockage centralisé des modèles dans un “model registry”.

Tester le serving MLFlow.

 

J’ai également rassemblé des conseils pour vous aider à vous lancer dans ce projet !

 

Concernant l’élaboration du modèle soyez vigilant sur deux points spécifiques au contexte métier : 

Le déséquilibre entre le nombre de bons et de moins bons clients doit être pris en compte pour élaborer un modèle pertinent, avec une méthode au choix.

Le déséquilibre du coût métier entre un faux négatif (FN - mauvais client prédit bon client : donc crédit accordé et perte en capital) et un faux positif (FP - bon client prédit mauvais : donc refus crédit et manque à gagner en marge).

Vous pourrez supposer, par exemple, que le coût d’un FN est dix fois supérieur au coût d’un FP.

Vous créerez un score “métier” (minimisation du coût d’erreur de prédiction des FN et FP) pour comparer les modèles, afin de choisir le meilleur modèle et ses meilleurs hyperparamètres. Attention cette minimisation du coût métier doit passer par l’optimisation du seuil qui détermine, à partir d’une probabilité, la classe 0 ou 1 (un “predict” suppose un seuil à 0.5 qui n’est pas forcément l’optimum).

En parallèle, maintenez pour comparaison et contrôle des mesures plus techniques, telles que l’AUC et l’accuracy.

D’autre part je souhaite que vous mettiez en œuvre une démarche d’élaboration des modèles avec Cross-Validation et optimisation des hyperparamètres, via GridsearchCV ou équivalent.

Un dernier conseil : si vous obtenez des scores supérieurs au 1er du challenge Kaggle (AUC > 0.82), posez-vous la question si vous n’avez pas de l’overfitting dans votre modèle !

 

Bon courage !

 

Michaël

Pièce-jointe :

Liste des outils MLOps à utiliser

 

Il est temps de plonger dans le travail ! Vous pouvez suivre les étapes ci-dessous pour vous guider.

Étapes
Vous allez découvrir et préparer les données nécessaires à la construction de votre modèle de scoring. Cela inclut le nettoyage, la fusion des différentes sources, l'encodage des variables et la création de nouvelles features pertinentes. L'objectif est de constituer un dataset propre et enrichi, prêt pour l'entraînement. Vous devrez aussi analyser la qualité de vos variables et les déséquilibres dans les classes.

 

Prérequis

Avoir exploré les données brutes fournies.

Avoir vérifié les formats et les valeurs manquantes.

Avoir identifié les colonnes clés pour les jointures.

Avoir pris en compte les enjeux métiers (par ex. déséquilibre des classes).

Résultat attendu

 Un jeu de données propre, fusionné et enrichi, prêt à être utilisé pour l'entraînement.

Recommandations

 

Charger chaque fichier séparément et inspecter ses colonnes.

Utiliser Pandas pour fusionner les jeux de données.

Visualiser la distribution des classes cibles.

Créer de nouvelles features à partir des variables existantes si nécessaire.

Éviter de supprimer trop rapidement les colonnes avec des valeurs manquantes : explorer les possibilités d’imputation.

 

Points de vigilance

Oublier de vérifier les doublons.

Supprimer des colonnes sans analyser leur importance métier.

Imputer sans documenter ni justifier.

Fusionner sans gérer les duplications ou pertes de lignes.

Encoder sans faire attention au type de modèle prévu (ordinal vs. nominal).


Outils

Pandas

Matplotlib et Seaborn pour la visualisation

Scikit-learn pour le preprocessing

Missingno pour visualiser les valeurs manquantes

Vous allez tracer vos expériences de modélisation avec MLflow : métriques, hyperparamètres, versions de modèles, etc. Vous utiliserez l’interface web pour visualiser vos runs et comparer les modèles.

 

Prérequis

Avoir installé MLflow.

Avoir configuré votre projet localement.

Résultat attendu

 

Des runs visibles dans l’UI MLflow avec les paramètres testés et les scores obtenus.

 

Recommandations

Commencer par intégrermlflow.start_run()dans vos notebooks.

Logger les métriques et les paramètres principaux.

Utilisermlflow.autolog()si vous utilisez des modèles compatibles.

Activer l’UI avecmlflow uipour visualiser les résultats.

Points de vigilance

Lancer MLflow sans environnement isolé : cela peut créer des conflits de versions de bibliothèques. Utiliser un environnement virtuel.

Oublier d’annoter les expériences (tags, noms, commentaires) : il serait difficile ensuite de comprendre les résultats dans l’interface MLflow.

Ne pas versionner les modèles enregistrés empêche de reproduire les résultats et de gérer leur cycle de vie.

Sauvegarder des fichiers inutiles ou trop volumineux dans MLflow encombre le système et ralentit l’interface.


Outils

MLFlow

Vous allez entraîner différents modèles de classification et comparer leurs performances sur des métriques métiers et classiques. L’objectif est de tester plusieurs familles de modèles (forêts, boosting, MLP, etc.) et de construire une première version de votre pipeline d’apprentissage. Vous devez aussi intégrer une validation croisée pour évaluer leur robustesse.

 

Prérequis

Avoir préparé un dataset propre et prêt à l’entraînement.

Avoir compris la nature déséquilibrée du jeu de données.

Avoir identifié les variables cibles et explicatives.

Avoir installé les bibliothèques de machine learning nécessaires.

Avoir paramétré MLFlow.


Résultat attendu


Un ou plusieurs modèles entraînés, avec validation croisée et premières métriques d’évaluation.

 

Recommandations

Commencer par tester des modèles simples (Logistic Regression, Random Forest).

Comparer ensuite avec des modèles plus puissants (XGBoost, LightGBM, MLP).

UtiliserStratifiedKFoldpour conserver la distribution de classes et pour garantir une évaluation robuste.

Entraîner les modèles dans des notebooks clairs et documentés.

Stocker les scores et les hyperparamètres testés.

Points de vigilance

Ne pas tester sans validation croisée.

Évaluer un modèle uniquement sur un split "train/test" unique peut donner des résultats très variables selon le hasard du découpage. Le risque est d’avoir une mauvaise estimation de la performance réelle et d’avoir un choix de modèle performant sur un split mais mauvais en généralisation.

Ne pas comparer des modèles sur des métriques inadaptées. Il vaut mieux utiliser Utiliser des métriques adaptées comme :

AUC-ROC,

Recall sur la classe minoritaire,

F1-score,

Coût métier personnalisé (FN > FP).

Oublier la stratification sur les classes cibles : en effet, les algorithmes peuvent être biaisés vers la classe majoritaire si le dataset contient beaucoup plus de bons clients que de mauvais. Utilisez la pondération des classes. 

Ne pas gérer le déséquilibre des classes biaise l’apprentissage. Utiliser unclass_weightou appliquer du sur-échantillonnage comme SMOTE.

Outils

Scikit-learn

XGBoost

LightGBM

Vous allez optimiser les hyperparamètres des modèles pour maximiser leurs performances selon des critères métier. Vous définirez aussi un seuil de décision optimal basé sur le coût des erreurs. L’objectif est de minimiser le coût métier total (FN > FP).

 

Prérequis

Avoir entraîné plusieurs modèles.

Avoir comparé leurs performances de base.

Avoir compris la notion de coût d’erreur.

Avoir défini une fonction de coût métier.


Résultat attendu

 

Un modèle avec hyperparamètres optimisés et seuil métier ajusté.

 

Recommandations

UtiliserGridSearchCVouOptunapour l’optimisation.

Définir une fonction de coût pondérant FN et FP.

Tester différents seuils de classification (de 0.1 à 0.9).

Tracer la courbe coût vs. seuil pour choisir le meilleur.

Points de vigilance

Garder le seuil par défaut (0.5) sans justification. Ce seuil ne reflète pas nécessairement les enjeux métiers. Il faut optimiser le seuil en fonction du coût FN vs FP.

Oublier de tracer le score métier en fonction du seuil : cela empêche d’identifier la meilleure décision.

Optimiser uniquement sur l’AUC ou l’accuracy.

Oublier d’adapter les métriques au business.

Choisir un modèle sans tester sa robustesse.

Outils

Scikit-learn GridSearchCV

Optuna

Pour vérifier que vous n’avez rien oublié dans la réalisation de votre exercice, téléchargez et complétez la fiche d’autoévaluation 

Parlez-en avec votre mentor durant votre dernière session de mentorat.