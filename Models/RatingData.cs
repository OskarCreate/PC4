using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Data;

using Microsoft.ML.Data;

namespace PC4.Models
{
    public class RatingData
    {
        [LoadColumn(0)]
        public float userId;

        [LoadColumn(1)]
        public float movieId;

        [LoadColumn(2)]
        public float Label;
    }

    public class RatingPrediction
    {
        public float Score;
    }
}
