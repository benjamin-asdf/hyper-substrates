(ns animismic.lib.particles
  (:require
   [bennischwerdtner.pyutils :as pyutils :refer
    [*torch-device*]]
   [bennischwerdtner.hd.binary-sparse-segmented :as hd]
   [tech.v3.datatype.functional :as f]
   [tech.v3.tensor :as dtt]
   [fastmath.random :as fm.rand]
   [fastmath.core :as fm]
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]))

;;
;;
(defprotocol ParticleField
  (state [this]))
;;
;;

(do
  ;;
  ;; Anything backed by a :native-buffer has a zero
  ;; copy pathway to and from numpy.
  ;; Https://clj-python.github.io/libpython-clj/Usage.html
  (alter-var-root #'hd/default-opts
                  (fn [m]
                    (assoc m
                           :tensor-opts {:container-type
                                         :native-heap})))
  (require-python '[numpy :as np])
  (require-python '[torch :as torch])
  (require-python '[torch.sparse :as torch.sparse])
  (require '[libpython-clj2.python.np-array]))


(defn brownian-update [])




;; ------------------------
;;
;;
;;
;;

(def field-size [30 30])
(torch/randn field-size :device pyutils/*torch-device*)





(def particles
  (torch/ge
   (torch/rand 10 :device pyutils/*torch-device*)
   ;; density
   0.5))


;; i->j
;; directed graph with geometry:
;; all i->j edges around each i
;;
;;
;;
;;       . . .
;;       . p .
;;       . . .
;;
;;

;; return true, if j is adjacent to i

(defn hamming-distance [p1 p2]
  (f/sum (f/not-eq p1 p2)))

(hamming-distance [0 1 0] [0 1 1])
(hamming-distance [0 1 0] [0 1 1])

(defn l-infnity-dist
  [p1 p2]
  (fm/abs (fm/- (f/reduce-max p1) (f/reduce-max p2))))

(defn local-geometry [p1 p2 dist]
  (<= (l-infnity-dist p1 p2) dist))

(l-infnity-dist [10 0] [0 0])
(l-infnity-dist [0 0] [10 0])

(local-geometry [0 0] [0 0] 1)
(local-geometry [0 0] [0 1] 1)
(local-geometry [0 0] [1 1] 1)
(local-geometry [0 0] [1 2] 1)


(defn local-geometry-1d [i j]
  (<= 1 (abs (- i j))))

(torch/tril_indices
 (torch/randn [3 3]))

(dtt/compute-tensor
 [3 3]
 (comp {true 1.0 false 0.0} local-geometry-2d)
 :float32)

(let
    [grid-size [3 3]]
  (dtt/compute-tensor
   grid-size
   (comp
    {false 0.0 true 1.0}
    (fn [i j]
      [(fm/rem i (grid-size 0))
       (fm/mod j (grid-size 0))]))
   :float32))


(let [grid-size [3 5]]
  (dtt/compute-tensor
   grid-size
   (fn [i j]
     [
      ;; p1:
      [(fm/rem i (grid-size 0))
       (fm/mod i (grid-size 0))]

      ;; p2:
      [(fm/rem j (grid-size 0))
       (fm/mod j (grid-size 0))]])
   :object))



(let [grid-size [3 3]]
  (dtt/compute-tensor
   grid-size
   (fn [i j]
     [
      ;; p1:
      ;; p2:
      [(fm/rem j (grid-size 0))
       (fm/mod j (grid-size 0))]

      [(fm/rem i (grid-size 0))
       (fm/mod i (grid-size 0))]])
   :object))





















(defn field-matrix [field-size]
  ;; i->j, directed graph with geometry around each i


  (torch/randn [3 3]))
