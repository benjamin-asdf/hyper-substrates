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
  (require-python '[local_square_matrix :as lsm])
  (require '[libpython-clj2.python.np-array]))

;;
;;
;; (defprotocol ParticleField
;;   (state [this])
;;   (update [this])
;;   (append-activations [this activations]))



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

;; reference impl
(defn local-square-matrix
  [grid-width]
  (let [->point (fn [i] [(fm/mod i grid-width)
                         (fm/quot i grid-width)])]
    (dtt/compute-tensor
      [(* grid-width grid-width) (* grid-width grid-width)]
      (fn [i j]
        (if (local-geometry (->point i) (->point j) 2)
          1.0
          0.0))
      :float32)))

(defn field-matrix
  [field-size]
  ;; i->j, directed graph with geometry around each i
  (lsm/local_square_matrix_torch field-size))

;; --------------

(defn grid-field
  [size update-fns]
  {:activations (torch/zeros [(* size size)] :dtype torch/float)
   :t 0
   :size size
   :update-fns update-fns
   :weights (field-matrix size)})

;; (defn append-activations-1 [state activations])

(defn normalize-weights
  [weights]
  (py/with-manual-gil-stack-rc-context
    (torch/div weights (torch/sum weights) :out weights))
  weights)

(defn normalize-update
  [{:as s :keys [t size]}]
  (if-not (zero? (mod t 5))
    s
    (assoc s :weights (field-matrix size))))

(defn update-grid-field
  [{:as s :keys [activations weights update-fns]}]
  (let [s (update s :t inc)
        s (reduce (fn [s op] (op s)) s update-fns)]
    s))

(defn read-particles [{:keys [activations size]}]
  (py.. activations (view size size)))

;; (defn default-update
;;   [weights activations]
;;   (let [out (torch/zeros_like activations
;;                               :device
;;                               pyutils/*torch-device*)]
;;     (py/with-gil-stack-rc-context
;;       (let [inputs (torch/mv weights activations)
;;             idxs (py.. (torch/topk
;;                         inputs
;;                         (long (py.. (torch/sum activations)
;;                                     item)))
;;                        -indices)
;;             actv
;;             (py/set-item!
;;              (torch/zeros_like activations :device pyutils/*torch-device*)
;;              idxs
;;              1)]
;;         (py.. out (copy_ actv))))))

(defn default-update
  [weights activations]
  (py/with-manual-gil-stack-rc-context
    (let [inputs (torch/mv weights activations)
          idxs (py.. (torch/topk inputs
                                 (long (py.. (torch/sum
                                              activations)
                                         item)))
                 -indices)
          _ (py.. activations (fill_ 0))
          _ (py.. activations (index_fill_ 0 idxs 1))])
    activations))


;; -------

;; (defn brownian-motion
;;   [weights]
;;   (let [out (torch/zeros_like weights
;;                               :device
;;                               pyutils/*torch-device*)]
;;     (py/with-gil-stack-rc-context
;;       (let [w (torch/mul weights
;;                          (torch/rand_like
;;                            weights
;;                            :device
;;                            pyutils/*torch-device*))]
;;         (py.. out (copy_ w))))))

(defn brownian-motion
  [weights]
  (py/with-gil-stack-rc-context
    (torch/mul weights
               (torch/rand_like weights
                                :device
                                pyutils/*torch-device*)
               :out
               weights))
  weights)

(comment
  (def weights (torch/rand [3 3]))
  (torch/mul weights
             (torch/zeros_like weights)
             :out
             weights))

(defn brownian-update
  [{:as state :keys [weights activations]}]
  (assoc state
         :activations
         (default-update
          (brownian-motion weights)
          activations)))

;; --------

(defn wormhole-babble
  [weights factor]
  (py/with-manual-gil-stack-rc-context
    (torch/bitwise_or weights
                      (torch/le factor
                                (torch/rand_like weights))
                      :out
                      weights))
  weights)

(defn vacuum-babble
  [activations factor]
  (py/with-manual-gil-stack-rc-context
    (torch/add activations
               (torch/le (torch/rand_like activations)
                         factor)
               :out
               activations)
    (torch/clamp_max_ activations 1))
  activations)

(comment
  (torch/clamp_max_ (torch/tensor [0 2 1]) 1)
  (vacuum-babble (torch/zeros [3] :dtype torch/float) 0.5)

  )

(defn decay
  [activations factor]
  (py/with-manual-gil-stack-rc-context
    (torch/mul activations
               (torch/ge (torch/rand_like activations)
                         factor)
               :out
               activations))
  activations)

(defn decay-update
  [{:as s :keys [activations decay-factor]}]
  (update s :activations decay decay-factor))

(defn vacuum-babble-update
  [{:as s :keys [activations vacuum-babble-factor]}]
  (update s
          :activations
          vacuum-babble
          vacuum-babble-factor))

;; --------


;; attract
(defn attracted
  [weight & weights]
  (reduce (fn [w1 w2]
            ;; (normalize-weights
            ;;  (torch/add w1 w2))
            (normalize-weights
              (torch/multiply w1 (torch/add w2 10))))
    weight
    weights))




;; (defn intersect
;;   [weight & weights]
;;   (let [boost-factor 5]                 ; Adjust this value to
;;                                         ; increase the effect
;;     (normalize-weights
;;      (reduce
;;       (fn [w1 w2]
;;         (torch/pow
;;          (torch/multiply w1 w2)
;;          (/ 1 boost-factor)))
;;       weight
;;       weights))))



;; repel
(defn repeled [field-a & fields]
  (torch/subtract field-a fields))


;; intersection
;; union
;;

(defmulti interact (fn [op & _] op))

(defmethod interact :attracted
  [_ f & fields]
  (update f
          :weights
          (fn [w]
            (apply attracted w (map :weights fields)))))

(comment
  (interact :attracted
            {:weights (torch/tensor [0 2 0])}
            {:weights (torch/tensor [0 2 0])}))

(defn interaction-update
  [field field-map spec]
  field
  ;; (reduce
  ;;  (fn [field [op & fields-ids]]
  ;;    (apply interact op field (map field-map
  ;;    fields-ids)))
  ;;  field
  ;;  spec)
)



(comment
  (def spec [[:attracted :orange]])
  (interaction-update {:weights (torch/tensor [0 2 0])}
                      {:orange {:weights (torch/tensor
                                          [0 2 0])}}
                      spec))

;; --------
(comment
  (def f (grid-field 3 [brownian-update]))
  (field-matrix 30)

  (default-update
   (brownian-motion (field-matrix 3))
   (torch/tensor
    [1 0 0 0 0 0 0 0 0]
    :dtype torch/float))


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
    )))


(comment
  (def f
    (assoc (grid-field 3
                       [vacuum-babble-update decay-update
                        brownian-update])
           :decay-factor 0.5
           :vacuum-babble-factor 0.5))
  (update-grid-field f)
  (decay (torch/ones [9] :dtype torch/float) 0.9)
  (decay (torch/ones [9] :dtype torch/float) 0.1)
  (normalize-weights (field-matrix 3))
  (torch/mv
   (normalize-weights (field-matrix 3))
   (torch/tensor [1 0 0 0 0 0 0 0 0] :dtype torch/float))
  (torch/mv
   (field-matrix 3)
   (torch/tensor [1 0 0 0 0 0 0 0 0] :dtype torch/float))
  (default-update
   (brownian-motion (field-matrix 3))
   (torch/tensor [1 0 0 0 0 0 0 0 0] :dtype torch/float)))

;; ------------------------
;;
;;
;;
;;

(comment
  (local-square-matrix 3)
  (local-square-matrix 9)
  (time (local-square-matrix 30))
  (field-matrix 30)
  (py.. (torch/mv (field-matrix 3)
                  (py.. (torch/tensor [[0 1 0] [0 0 0]
                                       [0 0 0]]
                                      :dtype
                                      torch/float)
                    (view -1)))
    (view 3 3))
  (py.. (torch/mv (field-matrix 3)
                  (py.. (torch/tensor [[0 0 0] [0 1 0]
                                       [0 0 0]]
                                      :dtype
                                      torch/float)
                    (view 9)))
    (view 3 3))
  (let [inputs (torch/mv (field-matrix 3)
                         (py.. (torch/tensor
                                [[0 0 0] [0 1 0] [0 1 0]]
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
     :object))
  (require-python '[local_square_matrix :as lsm])
  (torch/allclose (lsm/local_square_matrix_torch 3)
                  (field-matrix 3))
  (torch/allclose (time (lsm/local_square_matrix_torch 30))
                  (time (field-matrix 10))))

(comment



  )
