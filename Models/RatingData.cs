using Microsoft.ML.Data;

namespace PC4.Models
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string Text { get; set; }

        [LoadColumn(1)]
        public bool Label { get; set; }
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }
        public float Score { get; set; }
    }

    public class RatingData
    {
        [LoadColumn(0)]
        public string UserId { get; set; }

        [LoadColumn(1)]
        public string ProductId { get; set; }

        [LoadColumn(2)]
        public float Label { get; set; } // por ejemplo: puntuación de 1 a 5
    }

    public class RatingPrediction
    {
        public float Score { get; set; }
    }
}

