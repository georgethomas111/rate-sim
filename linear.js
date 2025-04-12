function linearRegression(x, y) {
  const n = x.length;

  if (n !== y.length) {
    throw new Error("Input arrays must have the same length.");
  }

  const sumX = x.reduce((acc, val) => acc + val, 0);
  const sumY = y.reduce((acc, val) => acc + val, 0);
  const sumXY = x.reduce((acc, val, i) => acc + val * y[i], 0);
  const sumX2 = x.reduce((acc, val) => acc + val * val, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  return {
    slope,
    intercept,
    predict: (xVal) => slope * xVal + intercept
  };
}

// Example usage:
const x = [1, 2, 3, 4, 5];
const y = [2, 4, 5, 4, 5];

const model = linearRegression(x, y);

console.log("Slope:", model.slope);
console.log("Intercept:", model.intercept);
console.log("Prediction for x=6:", model.predict(6));

