{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Fraud Detection Kaggle.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eYEyZ8NCQ9W0",
        "colab_type": "text"
      },
      "source": [
        "# Detecção de Fraudes com Cartões de Crédito\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nV6IvIZ3Q9u4",
        "colab_type": "text"
      },
      "source": [
        "## Classificando dados anônimos de transações por cartão de crédito como fraudulentas ou genuínas"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FtRTwjiRRAjA",
        "colab_type": "text"
      },
      "source": [
        "Neste repo, foi analisado um dataset muito popular que trata de fraudes com cartões de crédito. O objetivo é classificar transações como fraudulentas ou genuínas.\n",
        "\n",
        "Tirado da página do dataset no [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud):\n",
        "\n",
        "\n",
        "*The datasets contains transactions made by credit cards in September 2013 by european cardholders.\n",
        "This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.*\n",
        "\n",
        "*It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.*\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jeWZS2WOOpTd",
        "colab_type": "text"
      },
      "source": [
        "# Importando bibliotecas e dados"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LvUtQeQA_g-R",
        "colab_type": "code",
        "outputId": "e179e3a5-f3f8-4c4b-c926-0daf9898ab73",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 122
        }
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "from sklearn.preprocessing import RobustScaler\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.utils import class_weight\n",
        "from sklearn.metrics import classification_report, confusion_matrix, recall_score\n",
        "from sklearn.pipeline import make_pipeline\n",
        "\n",
        "from mlxtend.classifier import EnsembleVoteClassifier\n",
        "\n",
        "from keras.wrappers.scikit_learn import KerasClassifier\n",
        "\n",
        "from tensorflow.keras.models import Sequential, load_model\n",
        "from tensorflow.keras.layers import Dense, Activation, Dropout\n",
        "from tensorflow.keras.optimizers import Adam\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "\n",
        "from xgboost import XGBClassifier"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.\n",
            "  import pandas.util.testing as tm\n",
            "/usr/local/lib/python3.6/dist-packages/sklearn/externals/six.py:31: FutureWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).\n",
            "  \"(https://pypi.org/project/six/).\", FutureWarning)\n",
            "Using TensorFlow backend.\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LzdU1A2G_ezg",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "df = pd.read_csv('https://clouda-datasets.s3.amazonaws.com/creditcard.csv.zip')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lHV1h8Ajzy6-",
        "colab_type": "text"
      },
      "source": [
        "# Visualização de Dados"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qYWAwiJ2z4mZ",
        "colab_type": "code",
        "outputId": "dbdaeea2-3e95-4cf2-b1a1-6bff5079e036",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 663
        }
      },
      "source": [
        "df.info()"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 284807 entries, 0 to 284806\n",
            "Data columns (total 31 columns):\n",
            " #   Column  Non-Null Count   Dtype  \n",
            "---  ------  --------------   -----  \n",
            " 0   Time    284807 non-null  float64\n",
            " 1   V1      284807 non-null  float64\n",
            " 2   V2      284807 non-null  float64\n",
            " 3   V3      284807 non-null  float64\n",
            " 4   V4      284807 non-null  float64\n",
            " 5   V5      284807 non-null  float64\n",
            " 6   V6      284807 non-null  float64\n",
            " 7   V7      284807 non-null  float64\n",
            " 8   V8      284807 non-null  float64\n",
            " 9   V9      284807 non-null  float64\n",
            " 10  V10     284807 non-null  float64\n",
            " 11  V11     284807 non-null  float64\n",
            " 12  V12     284807 non-null  float64\n",
            " 13  V13     284807 non-null  float64\n",
            " 14  V14     284807 non-null  float64\n",
            " 15  V15     284807 non-null  float64\n",
            " 16  V16     284807 non-null  float64\n",
            " 17  V17     284807 non-null  float64\n",
            " 18  V18     284807 non-null  float64\n",
            " 19  V19     284807 non-null  float64\n",
            " 20  V20     284807 non-null  float64\n",
            " 21  V21     284807 non-null  float64\n",
            " 22  V22     284807 non-null  float64\n",
            " 23  V23     284807 non-null  float64\n",
            " 24  V24     284807 non-null  float64\n",
            " 25  V25     284807 non-null  float64\n",
            " 26  V26     284807 non-null  float64\n",
            " 27  V27     284807 non-null  float64\n",
            " 28  V28     284807 non-null  float64\n",
            " 29  Amount  284807 non-null  float64\n",
            " 30  Class   284807 non-null  int64  \n",
            "dtypes: float64(30), int64(1)\n",
            "memory usage: 67.4 MB\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yuunVTTvz7Dr",
        "colab_type": "code",
        "outputId": "92f7a5ee-d392-4772-d82e-68902101cf4c",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 224
        }
      },
      "source": [
        "df.head()"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Time</th>\n",
              "      <th>V1</th>\n",
              "      <th>V2</th>\n",
              "      <th>V3</th>\n",
              "      <th>V4</th>\n",
              "      <th>V5</th>\n",
              "      <th>V6</th>\n",
              "      <th>V7</th>\n",
              "      <th>V8</th>\n",
              "      <th>V9</th>\n",
              "      <th>V10</th>\n",
              "      <th>V11</th>\n",
              "      <th>V12</th>\n",
              "      <th>V13</th>\n",
              "      <th>V14</th>\n",
              "      <th>V15</th>\n",
              "      <th>V16</th>\n",
              "      <th>V17</th>\n",
              "      <th>V18</th>\n",
              "      <th>V19</th>\n",
              "      <th>V20</th>\n",
              "      <th>V21</th>\n",
              "      <th>V22</th>\n",
              "      <th>V23</th>\n",
              "      <th>V24</th>\n",
              "      <th>V25</th>\n",
              "      <th>V26</th>\n",
              "      <th>V27</th>\n",
              "      <th>V28</th>\n",
              "      <th>Amount</th>\n",
              "      <th>Class</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.0</td>\n",
              "      <td>-1.359807</td>\n",
              "      <td>-0.072781</td>\n",
              "      <td>2.536347</td>\n",
              "      <td>1.378155</td>\n",
              "      <td>-0.338321</td>\n",
              "      <td>0.462388</td>\n",
              "      <td>0.239599</td>\n",
              "      <td>0.098698</td>\n",
              "      <td>0.363787</td>\n",
              "      <td>0.090794</td>\n",
              "      <td>-0.551600</td>\n",
              "      <td>-0.617801</td>\n",
              "      <td>-0.991390</td>\n",
              "      <td>-0.311169</td>\n",
              "      <td>1.468177</td>\n",
              "      <td>-0.470401</td>\n",
              "      <td>0.207971</td>\n",
              "      <td>0.025791</td>\n",
              "      <td>0.403993</td>\n",
              "      <td>0.251412</td>\n",
              "      <td>-0.018307</td>\n",
              "      <td>0.277838</td>\n",
              "      <td>-0.110474</td>\n",
              "      <td>0.066928</td>\n",
              "      <td>0.128539</td>\n",
              "      <td>-0.189115</td>\n",
              "      <td>0.133558</td>\n",
              "      <td>-0.021053</td>\n",
              "      <td>149.62</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.0</td>\n",
              "      <td>1.191857</td>\n",
              "      <td>0.266151</td>\n",
              "      <td>0.166480</td>\n",
              "      <td>0.448154</td>\n",
              "      <td>0.060018</td>\n",
              "      <td>-0.082361</td>\n",
              "      <td>-0.078803</td>\n",
              "      <td>0.085102</td>\n",
              "      <td>-0.255425</td>\n",
              "      <td>-0.166974</td>\n",
              "      <td>1.612727</td>\n",
              "      <td>1.065235</td>\n",
              "      <td>0.489095</td>\n",
              "      <td>-0.143772</td>\n",
              "      <td>0.635558</td>\n",
              "      <td>0.463917</td>\n",
              "      <td>-0.114805</td>\n",
              "      <td>-0.183361</td>\n",
              "      <td>-0.145783</td>\n",
              "      <td>-0.069083</td>\n",
              "      <td>-0.225775</td>\n",
              "      <td>-0.638672</td>\n",
              "      <td>0.101288</td>\n",
              "      <td>-0.339846</td>\n",
              "      <td>0.167170</td>\n",
              "      <td>0.125895</td>\n",
              "      <td>-0.008983</td>\n",
              "      <td>0.014724</td>\n",
              "      <td>2.69</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1.0</td>\n",
              "      <td>-1.358354</td>\n",
              "      <td>-1.340163</td>\n",
              "      <td>1.773209</td>\n",
              "      <td>0.379780</td>\n",
              "      <td>-0.503198</td>\n",
              "      <td>1.800499</td>\n",
              "      <td>0.791461</td>\n",
              "      <td>0.247676</td>\n",
              "      <td>-1.514654</td>\n",
              "      <td>0.207643</td>\n",
              "      <td>0.624501</td>\n",
              "      <td>0.066084</td>\n",
              "      <td>0.717293</td>\n",
              "      <td>-0.165946</td>\n",
              "      <td>2.345865</td>\n",
              "      <td>-2.890083</td>\n",
              "      <td>1.109969</td>\n",
              "      <td>-0.121359</td>\n",
              "      <td>-2.261857</td>\n",
              "      <td>0.524980</td>\n",
              "      <td>0.247998</td>\n",
              "      <td>0.771679</td>\n",
              "      <td>0.909412</td>\n",
              "      <td>-0.689281</td>\n",
              "      <td>-0.327642</td>\n",
              "      <td>-0.139097</td>\n",
              "      <td>-0.055353</td>\n",
              "      <td>-0.059752</td>\n",
              "      <td>378.66</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1.0</td>\n",
              "      <td>-0.966272</td>\n",
              "      <td>-0.185226</td>\n",
              "      <td>1.792993</td>\n",
              "      <td>-0.863291</td>\n",
              "      <td>-0.010309</td>\n",
              "      <td>1.247203</td>\n",
              "      <td>0.237609</td>\n",
              "      <td>0.377436</td>\n",
              "      <td>-1.387024</td>\n",
              "      <td>-0.054952</td>\n",
              "      <td>-0.226487</td>\n",
              "      <td>0.178228</td>\n",
              "      <td>0.507757</td>\n",
              "      <td>-0.287924</td>\n",
              "      <td>-0.631418</td>\n",
              "      <td>-1.059647</td>\n",
              "      <td>-0.684093</td>\n",
              "      <td>1.965775</td>\n",
              "      <td>-1.232622</td>\n",
              "      <td>-0.208038</td>\n",
              "      <td>-0.108300</td>\n",
              "      <td>0.005274</td>\n",
              "      <td>-0.190321</td>\n",
              "      <td>-1.175575</td>\n",
              "      <td>0.647376</td>\n",
              "      <td>-0.221929</td>\n",
              "      <td>0.062723</td>\n",
              "      <td>0.061458</td>\n",
              "      <td>123.50</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>2.0</td>\n",
              "      <td>-1.158233</td>\n",
              "      <td>0.877737</td>\n",
              "      <td>1.548718</td>\n",
              "      <td>0.403034</td>\n",
              "      <td>-0.407193</td>\n",
              "      <td>0.095921</td>\n",
              "      <td>0.592941</td>\n",
              "      <td>-0.270533</td>\n",
              "      <td>0.817739</td>\n",
              "      <td>0.753074</td>\n",
              "      <td>-0.822843</td>\n",
              "      <td>0.538196</td>\n",
              "      <td>1.345852</td>\n",
              "      <td>-1.119670</td>\n",
              "      <td>0.175121</td>\n",
              "      <td>-0.451449</td>\n",
              "      <td>-0.237033</td>\n",
              "      <td>-0.038195</td>\n",
              "      <td>0.803487</td>\n",
              "      <td>0.408542</td>\n",
              "      <td>-0.009431</td>\n",
              "      <td>0.798278</td>\n",
              "      <td>-0.137458</td>\n",
              "      <td>0.141267</td>\n",
              "      <td>-0.206010</td>\n",
              "      <td>0.502292</td>\n",
              "      <td>0.219422</td>\n",
              "      <td>0.215153</td>\n",
              "      <td>69.99</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Time        V1        V2        V3  ...       V27       V28  Amount  Class\n",
              "0   0.0 -1.359807 -0.072781  2.536347  ...  0.133558 -0.021053  149.62      0\n",
              "1   0.0  1.191857  0.266151  0.166480  ... -0.008983  0.014724    2.69      0\n",
              "2   1.0 -1.358354 -1.340163  1.773209  ... -0.055353 -0.059752  378.66      0\n",
              "3   1.0 -0.966272 -0.185226  1.792993  ...  0.062723  0.061458  123.50      0\n",
              "4   2.0 -1.158233  0.877737  1.548718  ...  0.219422  0.215153   69.99      0\n",
              "\n",
              "[5 rows x 31 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HmFwDjOoz8Ap",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "df.drop('Time', axis=1, inplace=True)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bzn5ux0yeWoy",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        },
        "outputId": "517e2fbc-7013-4ee8-def6-62e7b5eb7eaf"
      },
      "source": [
        "df['Class'].value_counts()"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    284315\n",
              "1       492\n",
              "Name: Class, dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8GkZIVEbz8WW",
        "colab_type": "code",
        "outputId": "3615b06f-4a81-4d08-b917-f87b006ba4dc",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        }
      },
      "source": [
        "df['Class'].value_counts(normalize=True)*100"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    99.827251\n",
              "1     0.172749\n",
              "Name: Class, dtype: float64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bw6FYen8IND5",
        "colab_type": "code",
        "outputId": "e0cfeecd-11bd-400c-b87b-d5906d2ee9ba",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        }
      },
      "source": [
        "plt.figure(figsize=(12,6))\n",
        "sns.heatmap(df[df['Class']==1].corr(), cmap='viridis')"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f0c4f049860>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 8
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 864x432 with 2 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "66WWMaYS0EOP",
        "colab_type": "code",
        "outputId": "f9b7efe9-39c9-4356-f418-28fe58609e8c",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 420
        }
      },
      "source": [
        "df.corr()['Class'].drop('Class', axis = 0).plot(kind=\"bar\", figsize=(12,6))"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f0c4b1f1f60>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 864x432 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7X9NcPirUxr6",
        "colab_type": "text"
      },
      "source": [
        "# Dividindo dataset e treinando o modelo"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Byo87ytkUoTm",
        "colab_type": "text"
      },
      "source": [
        "A métrica utilizada para classificar os modelos foi o recall com cálculo \"macro\". Assim, é colocado um peso maior sobre a ocorrência de transações fraudulentas."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-ATaFmaAAFrG",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "X = df.drop('Class', axis=1)\n",
        "y = df['Class']\n",
        "X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1948)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nOBLXGDbXz_d",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "scaler = RobustScaler()\n",
        "\n",
        "X_train = scaler.fit_transform(X_train)\n",
        "X_test = scaler.transform(X_test)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ye7eYrBSGw9B",
        "colab_type": "text"
      },
      "source": [
        "Primeiramente serão treinados três modelos: redes neurais artificiais, regressão logística e XGgboost. Depois, os três modelos são reunidos em um mecanismo de votação para melhorar o score."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_EydvUwgghq_",
        "colab_type": "text"
      },
      "source": [
        "# ANN"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pLdMhZe6ZSR-",
        "colab_type": "code",
        "outputId": "64b9e84d-c743-409b-c57d-14dd1375ca5a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        }
      },
      "source": [
        "def ANN_model(n_units, n_layers, dropout_rate, decrease_rate):\n",
        "  model = Sequential()\n",
        "  for layer in range(n_layers):\n",
        "    model.add(Dense(units=n_units,activation='relu'))\n",
        "    model.add(Dropout(dropout_rate))\n",
        "    n_units = round(n_units*decrease_rate)   \n",
        "  model.add(Dense(units=1,activation='sigmoid'))\n",
        "  model.compile(optimizer='adam',\n",
        "                loss='binary_crossentropy',\n",
        "                metrics=['accuracy'])\n",
        "  return model\n",
        "\n",
        "skweight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)\n",
        "kerasweight = {}\n",
        "\n",
        "for weight in range(len(skweight)):\n",
        "  kerasweight[weight] = skweight[weight]\n",
        "\n",
        "callbacks = EarlyStopping(monitor='val_loss', patience=4)\n",
        "\n",
        "parameters = [{'kerasclassifier__n_units': [28, 14],\n",
        "               'kerasclassifier__n_layers':[2, 3],\n",
        "               'kerasclassifier__dropout_rate':[0.1, 0.3],\n",
        "               'kerasclassifier__decrease_rate':[0.8, 0.5],\n",
        "               'kerasclassifier__epochs':[15],\n",
        "               'kerasclassifier__verbose':[10],\n",
        "               }]\n",
        "\n",
        "pipe = make_pipeline(RobustScaler(),\n",
        "                     KerasClassifier(build_fn=ANN_model))\n",
        "\n",
        "grid_ann = GridSearchCV(estimator = pipe,\n",
        "                        param_grid = parameters,\n",
        "                        scoring='recall_macro',\n",
        "                        cv=5,\n",
        "                        n_jobs=-1,\n",
        "                        verbose=50,\n",
        "                        refit=True)\n",
        "\n",
        "grid_ann.fit(X_train, y_train, **{'kerasclassifier__validation_data':(X_test, y_test),\n",
        "                                  'kerasclassifier__callbacks':[callbacks],\n",
        "                                  'kerasclassifier__verbose':10,\n",
        "                                  'kerasclassifier__class_weight':kerasweight})\n",
        "\n",
        "# Getting predictions metrics\n",
        "predictions = (grid_ann.predict(X_test) > 0.5).astype(\"int32\")\n",
        "print('\\n')\n",
        "print(\"Melhor modelo: \" + str(grid_ann.best_params_))\n",
        "print('\\n')\n",
        "print(classification_report(y_test,predictions))\n",
        "print('\\n')\n",
        "print(confusion_matrix(y_test,predictions))\n",
        "print(\"Recall Macro: \", str(recall_score(y_test, predictions, average='macro'))[:6])"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Fitting 5 folds for each of 16 candidates, totalling 80 fits\n",
            "[Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.\n",
            "[Parallel(n_jobs=-1)]: Done   1 tasks      | elapsed:  3.3min\n",
            "[Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:  3.7min\n",
            "[Parallel(n_jobs=-1)]: Done   3 tasks      | elapsed:  5.5min\n",
            "[Parallel(n_jobs=-1)]: Done   4 tasks      | elapsed:  7.5min\n",
            "[Parallel(n_jobs=-1)]: Done   5 tasks      | elapsed:  8.0min\n",
            "[Parallel(n_jobs=-1)]: Done   6 tasks      | elapsed: 11.1min\n",
            "[Parallel(n_jobs=-1)]: Done   7 tasks      | elapsed: 11.7min\n",
            "[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed: 13.8min\n",
            "[Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed: 15.5min\n",
            "[Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed: 16.9min\n",
            "[Parallel(n_jobs=-1)]: Done  11 tasks      | elapsed: 19.1min\n",
            "[Parallel(n_jobs=-1)]: Done  12 tasks      | elapsed: 19.5min\n",
            "[Parallel(n_jobs=-1)]: Done  13 tasks      | elapsed: 21.8min\n",
            "[Parallel(n_jobs=-1)]: Done  14 tasks      | elapsed: 22.2min\n",
            "[Parallel(n_jobs=-1)]: Done  15 tasks      | elapsed: 25.4min\n",
            "[Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed: 26.0min\n",
            "[Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed: 27.9min\n",
            "[Parallel(n_jobs=-1)]: Done  18 tasks      | elapsed: 30.1min\n",
            "[Parallel(n_jobs=-1)]: Done  19 tasks      | elapsed: 30.9min\n",
            "[Parallel(n_jobs=-1)]: Done  20 tasks      | elapsed: 32.3min\n",
            "[Parallel(n_jobs=-1)]: Done  21 tasks      | elapsed: 32.8min\n",
            "[Parallel(n_jobs=-1)]: Done  22 tasks      | elapsed: 34.6min\n",
            "[Parallel(n_jobs=-1)]: Done  23 tasks      | elapsed: 35.5min\n",
            "[Parallel(n_jobs=-1)]: Done  24 tasks      | elapsed: 36.9min\n",
            "[Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed: 37.3min\n",
            "[Parallel(n_jobs=-1)]: Done  26 tasks      | elapsed: 39.8min\n",
            "[Parallel(n_jobs=-1)]: Done  27 tasks      | elapsed: 40.8min\n",
            "[Parallel(n_jobs=-1)]: Done  28 tasks      | elapsed: 41.8min\n",
            "[Parallel(n_jobs=-1)]: Done  29 tasks      | elapsed: 44.3min\n",
            "[Parallel(n_jobs=-1)]: Done  30 tasks      | elapsed: 44.5min\n",
            "[Parallel(n_jobs=-1)]: Done  31 tasks      | elapsed: 46.7min\n",
            "[Parallel(n_jobs=-1)]: Done  32 tasks      | elapsed: 47.0min\n",
            "[Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed: 49.2min\n",
            "[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed: 51.1min\n",
            "[Parallel(n_jobs=-1)]: Done  35 tasks      | elapsed: 51.3min\n",
            "[Parallel(n_jobs=-1)]: Done  36 tasks      | elapsed: 54.3min\n",
            "[Parallel(n_jobs=-1)]: Done  37 tasks      | elapsed: 54.8min\n",
            "[Parallel(n_jobs=-1)]: Done  38 tasks      | elapsed: 57.0min\n",
            "[Parallel(n_jobs=-1)]: Done  39 tasks      | elapsed: 57.8min\n",
            "[Parallel(n_jobs=-1)]: Done  40 tasks      | elapsed: 60.2min\n",
            "[Parallel(n_jobs=-1)]: Done  41 tasks      | elapsed: 60.5min\n",
            "[Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed: 62.1min\n",
            "[Parallel(n_jobs=-1)]: Done  43 tasks      | elapsed: 62.6min\n",
            "[Parallel(n_jobs=-1)]: Done  44 tasks      | elapsed: 64.9min\n",
            "[Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed: 66.1min\n",
            "[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed: 66.9min\n",
            "[Parallel(n_jobs=-1)]: Done  47 tasks      | elapsed: 68.4min\n",
            "[Parallel(n_jobs=-1)]: Done  48 tasks      | elapsed: 69.9min\n",
            "[Parallel(n_jobs=-1)]: Done  49 tasks      | elapsed: 72.1min\n",
            "[Parallel(n_jobs=-1)]: Done  50 tasks      | elapsed: 73.6min\n",
            "[Parallel(n_jobs=-1)]: Done  51 tasks      | elapsed: 73.9min\n",
            "[Parallel(n_jobs=-1)]: Done  52 tasks      | elapsed: 77.5min\n",
            "[Parallel(n_jobs=-1)]: Done  53 tasks      | elapsed: 77.8min\n",
            "[Parallel(n_jobs=-1)]: Done  54 tasks      | elapsed: 78.9min\n",
            "[Parallel(n_jobs=-1)]: Done  55 tasks      | elapsed: 80.2min\n",
            "[Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed: 81.2min\n",
            "[Parallel(n_jobs=-1)]: Done  57 tasks      | elapsed: 83.2min\n",
            "[Parallel(n_jobs=-1)]: Done  58 tasks      | elapsed: 85.0min\n",
            "[Parallel(n_jobs=-1)]: Done  59 tasks      | elapsed: 85.5min\n",
            "[Parallel(n_jobs=-1)]: Done  60 tasks      | elapsed: 88.7min\n",
            "[Parallel(n_jobs=-1)]: Done  61 tasks      | elapsed: 89.1min\n",
            "[Parallel(n_jobs=-1)]: Done  62 tasks      | elapsed: 91.1min\n",
            "[Parallel(n_jobs=-1)]: Done  63 tasks      | elapsed: 91.7min\n",
            "[Parallel(n_jobs=-1)]: Done  64 tasks      | elapsed: 93.9min\n",
            "[Parallel(n_jobs=-1)]: Done  65 tasks      | elapsed: 94.7min\n",
            "[Parallel(n_jobs=-1)]: Done  66 tasks      | elapsed: 97.6min\n",
            "[Parallel(n_jobs=-1)]: Done  67 tasks      | elapsed: 97.6min\n",
            "[Parallel(n_jobs=-1)]: Done  68 tasks      | elapsed: 99.6min\n",
            "[Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed: 100.3min\n",
            "[Parallel(n_jobs=-1)]: Done  70 tasks      | elapsed: 102.4min\n",
            "[Parallel(n_jobs=-1)]: Done  71 tasks      | elapsed: 102.8min\n",
            "[Parallel(n_jobs=-1)]: Done  72 tasks      | elapsed: 105.2min\n",
            "[Parallel(n_jobs=-1)]: Done  73 tasks      | elapsed: 105.9min\n",
            "[Parallel(n_jobs=-1)]: Done  74 tasks      | elapsed: 107.5min\n",
            "[Parallel(n_jobs=-1)]: Done  75 tasks      | elapsed: 109.0min\n",
            "[Parallel(n_jobs=-1)]: Done  76 tasks      | elapsed: 110.8min\n",
            "[Parallel(n_jobs=-1)]: Done  77 tasks      | elapsed: 111.3min\n",
            "[Parallel(n_jobs=-1)]: Done  80 out of  80 | elapsed: 117.0min finished\n",
            "Epoch 1/15\n",
            "Epoch 2/15\n",
            "Epoch 3/15\n",
            "Epoch 4/15\n",
            "Epoch 5/15\n",
            "Epoch 6/15\n",
            "Epoch 7/15\n",
            "Epoch 8/15\n",
            "\n",
            "\n",
            "Melhor modelo: {'kerasclassifier__decrease_rate': 0.8, 'kerasclassifier__dropout_rate': 0.1, 'kerasclassifier__epochs': 15, 'kerasclassifier__n_layers': 3, 'kerasclassifier__n_units': 14, 'kerasclassifier__verbose': 10}\n",
            "\n",
            "\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      0.97      0.99     56858\n",
            "           1       0.06      0.88      0.11       104\n",
            "\n",
            "    accuracy                           0.97     56962\n",
            "   macro avg       0.53      0.93      0.55     56962\n",
            "weighted avg       1.00      0.97      0.99     56962\n",
            "\n",
            "\n",
            "\n",
            "[[55386  1472]\n",
            " [   12    92]]\n",
            "Recall Macro:  0.9293\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "E7TYSeTSNyRN",
        "colab_type": "text"
      },
      "source": [
        "# Logistic Regression\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3-_7IPV0N1Nu",
        "colab_type": "code",
        "outputId": "ae13497e-c9fc-4ec7-ddb6-17ecd90e7ce8",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 513
        }
      },
      "source": [
        "LR_model = make_pipeline(RobustScaler(),\n",
        "                          LogisticRegression())\n",
        "\n",
        "parameters = [{'logisticregression__max_iter': [1000],\n",
        "               'logisticregression__C':[0.00001, 0.0001, 0.001, 0.1, 1],\n",
        "               'logisticregression__class_weight':['balanced', None]}]\n",
        "\n",
        "grid_LR = GridSearchCV(estimator = LR_model,\n",
        "                           param_grid = parameters,\n",
        "                           scoring = 'recall_macro',\n",
        "                           cv = 5,\n",
        "                           n_jobs = -1,\n",
        "                           verbose=10,\n",
        "                           refit=True)\n",
        "\n",
        "grid_LR.fit(X_train, y_train)\n",
        "\n",
        "# Prevendo com o melhor modelo\n",
        "y_pred = grid_LR.predict(X_test)\n",
        "# Criando matriz de confusão do melhor modelo\n",
        "cm =  confusion_matrix(y_test, y_pred)\n",
        "\n",
        "print(\"Melhor modelo: \" + str(grid_LR.best_params_))\n",
        "print('\\n')\n",
        "print(classification_report(y_test, y_pred))\n",
        "print('\\n')\n",
        "print(confusion_matrix(y_test, y_pred))\n",
        "print(\"Recall Macro: \", str(recall_score(y_test, y_pred, average='macro'))[:6])"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Fitting 5 folds for each of 10 candidates, totalling 50 fits\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "[Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.\n",
            "[Parallel(n_jobs=-1)]: Done   1 tasks      | elapsed:    1.6s\n",
            "[Parallel(n_jobs=-1)]: Done   4 tasks      | elapsed:    3.2s\n",
            "[Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:    7.4s\n",
            "[Parallel(n_jobs=-1)]: Done  14 tasks      | elapsed:   10.9s\n",
            "[Parallel(n_jobs=-1)]: Done  21 tasks      | elapsed:   17.7s\n",
            "[Parallel(n_jobs=-1)]: Done  28 tasks      | elapsed:   24.7s\n",
            "[Parallel(n_jobs=-1)]: Done  37 tasks      | elapsed:   45.3s\n",
            "[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:  1.2min\n",
            "[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:  1.3min finished\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Melhor modelo: {'logisticregression__C': 0.1, 'logisticregression__class_weight': 'balanced', 'logisticregression__max_iter': 1000}\n",
            "\n",
            "\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      0.98      0.99     56858\n",
            "           1       0.07      0.90      0.13       104\n",
            "\n",
            "    accuracy                           0.98     56962\n",
            "   macro avg       0.54      0.94      0.56     56962\n",
            "weighted avg       1.00      0.98      0.99     56962\n",
            "\n",
            "\n",
            "\n",
            "[[55615  1243]\n",
            " [   10    94]]\n",
            "Recall Macro:  0.9409\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JD47soXjRhCl",
        "colab_type": "text"
      },
      "source": [
        "# XGBOOST\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "u2m62bnXhIfO",
        "colab_type": "code",
        "outputId": "3f163d47-a044-4b69-9c13-11bc4a3d4b87",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 921
        }
      },
      "source": [
        "xgb_model = make_pipeline(RobustScaler(),\n",
        "                          XGBClassifier())\n",
        "\n",
        "parameters = {'xgbclassifier__learning_rate': [0.01, 0.1, 1],\n",
        "              'xgbclassifier__n_estimators': [300],\n",
        "              'xgbclassifier__max_depth': [7, 15],\n",
        "              'xgbclassifier__scale_pos_weight': [583],\n",
        "              'xgbclassifier__seed': [1948]}\n",
        "\n",
        "grid_xgb = GridSearchCV(estimator = xgb_model,\n",
        "                           param_grid = parameters,\n",
        "                           scoring = 'recall_macro',\n",
        "                           cv = 5,\n",
        "                           n_jobs = -1,\n",
        "                           verbose=50,\n",
        "                           refit=True)\n",
        "\n",
        "grid_xgb.fit(X_train, y_train)\n",
        "\n",
        "# Prevendo com o melhor modelo\n",
        "y_pred = grid_xgb.predict(X_test)\n",
        "# Criando matriz de confusão do melhor modelo\n",
        "cm =  confusion_matrix(y_test, y_pred)\n",
        "\n",
        "print(\"Melhor modelo: \" + str(grid_xgb.best_params_))\n",
        "print('\\n')\n",
        "print(classification_report(y_test, y_pred))\n",
        "print('\\n')\n",
        "print(pd.DataFrame(cm))\n",
        "print(\"Recall macro: \" + str(recall_score(y_test, y_pred))[:6])"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Fitting 5 folds for each of 6 candidates, totalling 30 fits\n",
            "[Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.\n",
            "[Parallel(n_jobs=-1)]: Done   1 tasks      | elapsed:  7.9min\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/joblib/externals/loky/process_executor.py:691: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.\n",
            "  \"timeout or by a memory leak.\", UserWarning\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "[Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:  8.2min\n",
            "[Parallel(n_jobs=-1)]: Done   3 tasks      | elapsed: 16.0min\n",
            "[Parallel(n_jobs=-1)]: Done   4 tasks      | elapsed: 16.2min\n",
            "[Parallel(n_jobs=-1)]: Done   5 tasks      | elapsed: 24.2min\n",
            "[Parallel(n_jobs=-1)]: Done   6 tasks      | elapsed: 29.2min\n",
            "[Parallel(n_jobs=-1)]: Done   7 tasks      | elapsed: 37.0min\n",
            "[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed: 40.5min\n",
            "[Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed: 49.7min\n",
            "[Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed: 53.2min\n",
            "[Parallel(n_jobs=-1)]: Done  11 tasks      | elapsed: 56.3min\n",
            "[Parallel(n_jobs=-1)]: Done  12 tasks      | elapsed: 59.6min\n",
            "[Parallel(n_jobs=-1)]: Done  13 tasks      | elapsed: 62.9min\n",
            "[Parallel(n_jobs=-1)]: Done  14 tasks      | elapsed: 66.3min\n",
            "[Parallel(n_jobs=-1)]: Done  15 tasks      | elapsed: 69.4min\n",
            "[Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed: 73.8min\n",
            "[Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed: 76.8min\n",
            "[Parallel(n_jobs=-1)]: Done  18 tasks      | elapsed: 81.0min\n",
            "[Parallel(n_jobs=-1)]: Done  19 tasks      | elapsed: 84.6min\n",
            "[Parallel(n_jobs=-1)]: Done  20 tasks      | elapsed: 87.0min\n",
            "[Parallel(n_jobs=-1)]: Done  21 tasks      | elapsed: 88.5min\n",
            "[Parallel(n_jobs=-1)]: Done  22 tasks      | elapsed: 89.5min\n",
            "[Parallel(n_jobs=-1)]: Done  23 tasks      | elapsed: 90.9min\n",
            "[Parallel(n_jobs=-1)]: Done  24 tasks      | elapsed: 91.9min\n",
            "[Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed: 93.3min\n",
            "[Parallel(n_jobs=-1)]: Done  26 tasks      | elapsed: 94.4min\n",
            "[Parallel(n_jobs=-1)]: Done  27 tasks      | elapsed: 95.7min\n",
            "[Parallel(n_jobs=-1)]: Done  28 out of  30 | elapsed: 96.9min remaining:  6.9min\n",
            "[Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed: 98.9min remaining:    0.0s\n",
            "[Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed: 98.9min finished\n",
            "Melhor modelo: {'xgbclassifier__learning_rate': 0.01, 'xgbclassifier__max_depth': 7, 'xgbclassifier__n_estimators': 300, 'xgbclassifier__scale_pos_weight': 583, 'xgbclassifier__seed': 1948}\n",
            "\n",
            "\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      1.00      1.00     56858\n",
            "           1       0.53      0.79      0.63       104\n",
            "\n",
            "    accuracy                           1.00     56962\n",
            "   macro avg       0.76      0.89      0.82     56962\n",
            "weighted avg       1.00      1.00      1.00     56962\n",
            "\n",
            "\n",
            "\n",
            "       0   1\n",
            "0  56785  73\n",
            "1     22  82\n",
            "Recall macro: 0.7884\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "evK3xvv_ppL_",
        "colab_type": "text"
      },
      "source": [
        "# Mecanismo de Votação (Ensemble Voting Classifier)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GbFnhMhmG-P5",
        "colab_type": "text"
      },
      "source": [
        "Curiosamente, o modelo mais simples, Regressão Logística, mostrou o melhor resultado. Depois de reunir os três modelos em um mecanismo de votação, o score ficou melhor ainda."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Mm3xjJNY-blS",
        "colab_type": "code",
        "outputId": "d04d6077-7ab0-4677-d7d6-24c83434ef95",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 306
        }
      },
      "source": [
        "eclf1 = EnsembleVoteClassifier(clfs=[grid_LR, grid_xgb, grid_ann], voting='soft', refit=False)\n",
        "eclf1.fit(X_train, y_train)\n",
        "\n",
        "# Prevendo com o melhor modelo\n",
        "y_pred = eclf1.predict(X_test)\n",
        "# Criando matriz de confusão do melhor modelo\n",
        "cm =  confusion_matrix(y_test, y_pred)\n",
        "\n",
        "print(\"Recall Macro: \" + str(recall_score(y_test, y_pred, average='macro'))[:6])\n",
        "print('\\n')\n",
        "print(classification_report(y_test, y_pred))\n",
        "print('\\n')\n",
        "print(pd.DataFrame(cm))"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Recall Macro: 0.9343\n",
            "\n",
            "\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      0.99      1.00     56858\n",
            "           1       0.20      0.88      0.33       104\n",
            "\n",
            "    accuracy                           0.99     56962\n",
            "   macro avg       0.60      0.93      0.66     56962\n",
            "weighted avg       1.00      0.99      1.00     56962\n",
            "\n",
            "\n",
            "\n",
            "       0    1\n",
            "0  56497  361\n",
            "1     13   91\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}