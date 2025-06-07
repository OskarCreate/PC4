using Microsoft.ML.Data;

public class SentimentData
{
    [LoadColumn(0)] public bool Label { get; set; }
    [LoadColumn(1)] public string Text { get; set; }
}

public class SentimentPrediction : SentimentData
{
    [ColumnName("PredictedLabel")] public bool Prediction { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
