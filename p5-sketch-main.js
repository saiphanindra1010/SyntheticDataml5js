const p5Main = new p5((s) => {

    const a = tf.scalar(s.random(-1, 1))
    s.a = tf.variable(a, true)
    a.dispose()
 

    const b = tf.scalar(s.random(-1, 1))
    s.b = tf.variable(b, true)
    b.dispose()
    const c = tf.scalar(s.random(-1, 1))
    s.c = tf.variable(c, true)
    c.dispose()

    const d = tf.scalar(s.random(-1, 1))
    s.d = tf.variable(d, true)
  
    d.dispose()

    const e = tf.scalar(s.random(-1, 1))
    s.e = tf.variable(e, true)
   
    e.dispose()

    const f = tf.scalar(s.random(-1, 1))
    s.f = tf.variable(f, true)

    f.dispose()
    s.predict = x => (
        tf.tidy(() => {
            const A = s.a.mul(x.pow(tf.scalar(5))) 
            const B = s.b.mul(x.pow(tf.scalar(4)))
            const C = s.c.mul(x.pow(tf.scalar(3))) 
            const D = s.d.mul(x.square())          
            const E = s.e.mul(x)                   
            const F = s.f                         
            return A.add(B).add(C).add(D).add(E).add(F)
        })
    )

    s.loss = (predictions, labels) => {
        const meanSquareError = predictions.sub(labels).square().mean()
        return meanSquareError
    }

    // s.optimizer = tf.train.sgd(0.1)
    s.optimizer = tf.train.adam()

    // p5.js setup -------------------------------------------------------------
    s.setup = () => {
        s.createCanvas(800, 800)
        //s.createCanvas(displayWidth/2, displayHeight);

        s.xs_data = []
        s.ys_data = []
        for (let i = 0; i <= 1; i += 0.01) {
            s.xs_data.push(i)
            s.ys_data.push(s.map(s.noise(i * 3 + s.frameCount) + s.random(-0.1, 0.1), 0, 1, -0.5, 0.5))
        }
        s.xs = tf.tensor1d(s.xs_data)
        s.ys = tf.tensor1d(s.ys_data)
    }
    // -------------------------------------------------------------------------

    s.draw = async () => {
        const opt = s.optimizer.minimize(
            () => {
                const ys_pred = s.predict(s.xs)
                return s.loss(ys_pred, s.ys)
            },
            true,
            
        )
        const opt_data = await opt.data()
        opt.dispose()

        await tf.nextFrame()

        const ys_pred = s.predict(s.xs)
        const ys_pred_data = await ys_pred.data()
        ys_pred.dispose()

        //s.background(0)
        s.background(131, 131, 131);
        s.drawGrid()
        s.drawCloud(s.xs_data, s.ys_data)
        s.drawPredCurve(s.xs_data, ys_pred_data)

        if (!(s.frameCount % 100) || s.frameCount === 1) {
            console.log('------------------')
            console.log('frame count:', s.frameCount)
            console.log('numTensors (in tidy): ', tf.memory().numTensors)
            console.log('data:', opt_data[0])
        }

        if (s.frameCount % 2345 === 0) {
            s.xs.dispose()
            s.ys.dispose()
            s.setup()
        }
    }
    // -------------------------------------------------------------------------
    s.drawCloud = (xs, ys) => {
        s.stroke(212, 212, 212)
        s.strokeWeight(6)
        ys.forEach((y, i) => s.point(xs[i] * s.width, s.height / 2 - y * s.height))
    }

    s.drawPredCurve = (xs, ys) => {
        s.beginShape()
        s.stroke(0, 0, 0)
        s.strokeWeight(4)
        s.noFill()
        ys.forEach((y, i) => s.vertex(xs[i] * s.width, s.height / 2 - y * s.height))
        s.endShape()
    }

    s.drawGrid = () => {
        s.stroke(255, 63)
        s.strokeWeight(1)
        for (let y = 0; y < s.height; y += s.height / 10) {
            s.line(0, y, s.width, y)
        }

        s.stroke(255, 31)
        for (let y = s.height / 20; y < s.height; y += s.height / 10) {
            s.line(0, y, s.width, y)
        }

        s.stroke(255, 63)
        for (let x = 0; x < s.height; x += s.height / 10) {
            s.line(x, 0, x, s.height)
        }

        s.stroke(255, 31)
        for (let x = s.height / 20; x < s.height; x += s.height / 10) {
            s.line(x, 0, x, s.height)
        }

        s.stroke(255, 127)
        s.strokeWeight(3)
        const originX = 0
        const originY = s.height / 2
        s.line(originX - s.width / 20, originY, originX + s.width / 20, originY)
        s.line(originX, originY - s.height / 20, originX, originY + s.height / 20)
    }
}, 'p5-main')

window.p5Main = p5Main
