%% ===============================================================
%  Entraînement de ResNet-18 avec affichage du graphe de learning
% ===============================================================

clc; clear; close all;

%% 1. Charger ResNet-18 pré-entraîné
net = resnet18;

%% 2. Charger ton jeu de données
dataDir = fullfile("dataset");
imdsTrain = imageDatastore(fullfile(dataDir,"train"), ...
    'IncludeSubfolders', true, 'LabelSource','foldernames');
imdsVal = imageDatastore(fullfile(dataDir,"val"), ...
    'IncludeSubfolders', true, 'LabelSource','foldernames');

numClasses = numel(categories(imdsTrain.Labels));
disp("Nombre de classes : " + numClasses);

%% 3. Redimensionner les images
inputSize = net.Layers(1).InputSize; % 224x224x3
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augimdsVal = augmentedImageDatastore(inputSize(1:2), imdsVal);

%% 4. Modifier les dernières couches pour ton nombre de classes
lgraph = layerGraph(net);

newLayers = [
    fullyConnectedLayer(numClasses, 'Name','fc_new', ...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
];

lgraph = replaceLayer(lgraph,'fc1000',newLayers(1));
lgraph = replaceLayer(lgraph,'prob',newLayers(2));
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newLayers(3));

%% 5. Définir les options d'entraînement
options = trainingOptions('sgdm', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',8, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsVal, ...
    'ValidationFrequency',20, ...
    'Verbose',true, ...
    'Plots','training-progress');  % <- C'est ici que le graphe de learning apparaît

%% 6. Lancer l'entraînement
[trainedNet, trainInfo] = trainNetwork(augimdsTrain, lgraph, options);

%% 7. Évaluer le réseau sur le jeu de validation
YPred = classify(trainedNet, augimdsVal);
YTrue = imdsVal.Labels;

accuracy = mean(YPred == YTrue);
disp("Précision sur le jeu de validation : " + accuracy);

%% 8. Sauvegarder le réseau
save trainedResNet18 trainedNet trainInfo
