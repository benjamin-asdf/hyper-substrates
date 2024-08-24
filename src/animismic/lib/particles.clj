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
  (state [this])
  (update [this])
  (append-activations [this activations]))

(def field-size [30 30])

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

(defn l-infnity-dist
  [p1 p2]
  (fm/abs (fm/- (f/reduce-max p1) (f/reduce-max p2))))

(defn l1-loss
  [p1 p2]
  (f/sum (f/abs (f/- p1 p2))))


(comment
  (l1-loss [0 2] [2 2])

  ;; . . x
  ;; . . .
  ;; . . p

  (l1-loss [1 2] [2 2])
  ;; . . .
  ;; . . x
  ;; . . p

  )



(defn local-geometry-1d [i j]
  (<= 1 (abs (- i j))))

(defn local-geometry [p1 p2 dist]
  (< 0 (l1-loss p1 p2) dist))

(defn local-square-matrix
  [grid-width]
  (let [->point (fn [i] [(fm/mod i grid-width)
                         (fm/quot i grid-width)])]
    (dtt/compute-tensor
      [(* grid-width grid-width) (* grid-width grid-width)]
      (fn [i j]
        (if
            (local-geometry (->point i) (->point j) 2)
            1.0
            0.0))
      :float32)))

(defn field-matrix
  [field-size]
  ;; i->j, directed graph with geometry around each i
  (pyutils/ensure-torch
   (local-square-matrix field-size)))

;; --------------

(defn grid-field
  [size update-fn]
  {:activations
   (torch/zeros [(* size size)] :dtype torch/float)
   :t 0
   :size size
   :update-fn update-fn
   :weights (field-matrix size)})

;; (defn append-activations-1 [state activations])

(defn update-grid-field
  [{:as s :keys [activations weights update-fn]}]
  (-> s
      (update :t inc)
      (assoc :activations (update-fn weights activations))))

(defn read-particles [{:keys [activations size]}]
  (py.. activations (view size size)))

;; -------

(defn brownian-motion
  [weights]
  (torch/mul weights
             (torch/rand_like weights
                              :device
                              pyutils/*torch-device*)))
(defn brownian-update
  [weights activations]
  (default-update (brownian-motion weights) activations))





;; --------

(defn default-update
  [weights activations]
  (let [inputs (torch/mv weights activations)
        idxs (py.. (torch/topk inputs
                               (long (py.. (torch/sum
                                            activations)
                                       item)))
               -indices)]
    (py/set-item! (torch/zeros_like activations) idxs 1)))


(def f
  (grid-field 3 brownian-update))




(comment
  (default-update
   (brownian-motion (field-matrix 3))
   (torch/tensor [1 0 0 0 0 0 0 0 0] :dtype torch/float))
  (update-grid-field
   (assoc
    (update-grid-field f)
    :activations
    (torch/tensor [1 0 0 0 0 0 0 0 0] :dtype torch/float)
    ;; (py.. (torch/ge (torch/rand [(* 3 3)]
    ;;                             :dtype
    ;;                             torch/float)
    ;;                 0.5)
    ;;   (to :dtype torch/float))
    ))

  )









;; ---------






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





;; ------------------------
;;
;;
;;
;;













(comment
  (py..
      (torch/mv
       (field-matrix 3)
       (py..
           (torch/tensor
            [[0 1 0]
             [0 0 0]
             [0 0 0]]
            :dtype
            torch/float)
           (view -1)))
      (view 3 3))

  (py..
      (torch/mv
       (field-matrix 3)
       (py..
           (torch/tensor
            [[0 0 0]
             [0 1 0]
             [0 0 0]]
            :dtype
            torch/float)
         (view 9)))
    (view 3 3))


  (let [inputs
        (torch/mv (field-matrix 3)
                  (py.. (torch/tensor [[0 0 0] [0 1 0]
                                       [0 1 0]]
                                      :dtype
                                      torch/float)
                    (view -1)))
        idxs (py.. (torch/topk inputs 2) -indices)]
    (py/set-item! (torch/zeros [9]) idxs 1)))

(comment
  (let [grid-width 3
        ->point (fn [i] [(fm/mod i grid-width)
                         (fm/quot i grid-width)])]
    (dtt/compute-tensor
     [(* grid-width grid-width) (* grid-width grid-width)]
     (fn [i j] [(l1-loss (->point i) (->point j)) :p1
                (->point i) :p2 (->point j) :i i :j j])
     :object)))
