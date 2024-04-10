package main

import (
	"fmt"
	"log"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/losses"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn/recurrent/lstm"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/optimizers"
	"github.com/nlpodyssey/spago/optimizers/sgd"
)

/*
	LSTM para prever o ultimo elemento de um serie temporal
	da temperatura do ar em graus celcius.
	No exemplo abaixo, será descoberto o valor do
	elemento da posição 24, ou seja, 27.21.
*/

type T = float64

func main() {
	data := []T{25.96, 27.39, 32.18, 30.51, 35.70, 33.92, 33.50, 29.22, 27.83, 26.15, 25.42, 24.57, 23.82, 23.34, 22.98, 22.77, 22.35, 22.23, 22.25, 22.23, 22.00, 22.78, 23.88, 25.50}

	model := lstm.New[T](len(data), len(data)).WithRefinedGates(true)

	x := mat.NewDense[T](mat.WithBacking(data), mat.WithGrad(true))

	strategy := sgd.New[T](sgd.NewConfig(0.0001, 0.09, true))
	optimizer := optimizers.New(nn.Parameters(model), strategy)

	y := mat.NewDense[T](mat.WithBacking(data))
	for epoch := 0; epoch < 10; epoch++ {	
		predictions := model.Forward(x)
		loss := losses.MSE(predictions[0], y, true)
		
		if err := ag.Backward(loss); err != nil {
			log.Fatal(err.Error())
		}
		if err := optimizer.Optimize(); err != nil {
			log.Fatal(err.Error())
		}
		if epoch%2 == 0 {
			fmt.Printf("Epoch %d, Loss: %.6f\n", epoch, loss.Value())
		}
	}
	predictions := model.Forward(x)
	fmt.Printf("Input:\n%.2f\nPrediction:\n%.2f\n", data, predictions[0].Value().Data())
}