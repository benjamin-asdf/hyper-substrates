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
  (require-python '[builtins :as builtins])
  (require '[libpython-clj2.python.np-array]))

(def excitability-threshold 1e-4)

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

(defn log-normal-field-matrix
  [field-size mu sigma]
  (let [base-matrix (lsm/local_square_matrix_torch field-size)
        log-normal-noise (torch/exp (torch/add (torch/mul (torch/randn_like base-matrix) sigma) mu))
        log-normal-matrix (torch/mul base-matrix log-normal-noise)]
    log-normal-matrix))

(defn distribute-log-normal
  [base-matrix mu sigma]
  (let [log-normal-noise
          (torch/exp
           (torch/add
            (torch/mul
             (torch/randn_like
              base-matrix
              :device
              pyutils/*torch-device*)
             sigma)
            mu))
        log-normal-matrix (torch/mul base-matrix
                                     log-normal-noise)]
    log-normal-matrix))

(defn adjacency-matrix
  [grid-size distance]
  (let [[x y] (torch/meshgrid
                (torch/arange grid-size
                              :device
                              pyutils/*torch-device*)
                (torch/arange grid-size
                              :device
                              pyutils/*torch-device*))
        positions (torch/stack [(py.. x (flatten))
                                (py.. y (flatten))]
                               :dim
                               1)
        diff (torch/subtract (torch/unsqueeze positions 1)
                             (torch/unsqueeze positions 0))
        toroidal_diff
          (torch/fmod (torch/add diff
                                 (torch/unsqueeze
                                   (torch/tensor
                                     [grid-size]
                                     :device
                                     pyutils/*torch-device*)
                                   0))
                      grid-size)
        toroidal_diff_min (torch/min toroidal_diff
                                     (torch/subtract
                                       grid-size
                                       toroidal_diff))
        distances
          (torch/sum (torch/abs toroidal_diff_min) :dim 2)
        adjacency (torch/le distances distance)
        adjacency (py.. adjacency (to :dtype torch/float))]
    adjacency))

(defn field-matrix
  [field-size]
  ;; i->j, directed graph with geometry around each i
  ;;
  ;; make it toroidal
  ;;
  ;; (pyutils/torch->jvm
  ;;  (py/get-item
  ;;   (distribute-log-normal (adjacency-matrix 5 1) 1
  ;;   1)
  ;;   [0]))
  (distribute-log-normal (adjacency-matrix field-size 2)
                         1
                         1))


;; --------------

;; Inhibitory interneurons

;; . . . .
;; . . . .
;; . X---|
;; . . . .
;;

;; idea 1:
;; Random negative weights in the matrix
;; (guess that would be generalization of hopfield nets)
;;
;; idea 2:
;; model basket cells with rand connections and their own activity
;;
;;
;; idea 3:
;; does a cap-k logic already implement this essentially?
;;
;;


;; Funny, this has similarities with Truing patterns.
;; P - makes more P and S locally (~ local activation)
;; S - diffuses faster than P and inhibits P (~ long range inhibition)
;;

;; --------------

(defn grid-field
  [size update-fns]
  (let [matrix (field-matrix size)]
    {:N (* size size)
     :activations (torch/zeros [(* size size)]
                               :dtype torch/float
                               :device
                               pyutils/*torch-device*)
     :excitability (torch/ones [(* size size)]
                               :dtype torch/float
                               :device
                               pyutils/*torch-device*)
     :field-matrix matrix
     :size size
     :t 0
     :update-fns update-fns
     :weights (py.. matrix (clone))}))

;; (defn append-activations-1 [state activations])

(defn normalize-weights
  [weights]
  (py/with-manual-gil-stack-rc-context
    (torch/div weights (torch/sum weights) :out weights))
  weights)

(defn reset-weights-update
  [{:as s :keys [t size field-matrix]}]
  (if-not (zero? (mod t 5))
    s
    (assoc s :weights (py.. field-matrix (clone)))))

(defn reset-excitability
  [{:as s :keys [t size field-matrix]}]
  (assoc s
         :excitability (torch/ones (py.. field-matrix (size 0))
                                   :device
                                   pyutils/*torch-device*)))

(defn reset-excitability-update
  [{:as s :keys [t]}]
  (if-not (zero? (mod t 20)) s (reset-excitability s)))

(defn update-grid-field
  [{:as s :keys [activations weights update-fns]}]
  (let [s (update s :t inc)
        s (reduce (fn [s op] (op s)) s update-fns)]
    s))

(defn read-particles [{:keys [activations size]}]
  (py.. activations (view size size)))

(defn read-activations [{:keys [activations]}]
  activations)

(defn read-hdv
  ([s] (read-hdv s hd/default-opts))
  ([{:keys [activations size]}
    {:bsdc-seg/keys [segment-length N]}]
   (py/set-item!
    (torch/zeros N :device pyutils/*torch-device*)
    (builtins/slice 0 (py.. activations (size 0)))
    activations)))

(defn rand-activations
  [N density]
  (py.. (torch/lt
          (torch/rand [N] :device pyutils/*torch-device*)
          density)
    (to :dtype torch/float)))

(comment
  (py/set-item!
   (py.. (torch/zeros 30) (view -1 10))
   (builtins/slice 0 10)
   (py.. (torch/arange 9) (view 3 3)))
  (py/set-item!
   (py.. (torch/zeros 30) (view -1 10))
   (builtins/slice 0 10)
   (py.. (torch/arange 9) (view 3 3))))

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

(defn current-inputs
  [{:keys [weights activations excitability]}]
  (let [inputs (torch/mv weights activations)
        inputs (torch/mul inputs excitability)]
    inputs))

(defn default-update
  [weights activations excitability]
  (py/with-manual-gil-stack-rc-context
    (let [inputs (torch/mv weights activations)
          inputs (torch/mul inputs excitability)
          idxs (py.. (torch/topk
                       inputs
                       (min (long (py.. (torch/sum
                                          activations)
                                        item))
                            (py.. activations (size 0))))
                     -indices)
          _ (py.. activations (fill_ 0))
          _ (py.. activations (index_fill_ 0 idxs 1))])
    activations))

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
  [{:as state :keys [weights activations excitability]}]
  (assoc state
    :activations (default-update (brownian-motion weights)
                                 activations
                                 excitability)))

;; -------------

(defn interact-inhibiting
  [field-producer field-inhibitor factor]
  (let [ ;;
        ;;
        ;; update activations of field-producer so that
        ;; there are fewer,
        ;;
        activations (:activations field-producer)
        idxs-killed
        (py.. (torch/topk
               (current-inputs field-inhibitor)
               (min (py.. activations (size 0))
                    (long (* factor
                             (py.. (torch/sum
                                    (:activations
                                     field-inhibitor))
                               (item))))))
          -indices)
        _ (py.. activations (index_fill_ 0 idxs-killed 0))])
  field-producer)


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
    (torch/add
     activations
     (torch/le (torch/rand_like activations) factor)
     :out
     activations)
    (torch/clamp_max_ activations 1))
  activations)

(defn production
  [field-target field-inputs production-factor]
  (let [activations (:activations field-target)
        production-factor
        (if (fn? production-factor)
          production-factor
          (fn [n] (* production-factor n)))
        idxs-fresh (py.. (torch/topk
                          (current-inputs field-inputs)
                          (min
                           (long (production-factor (py.. (torch/sum (:activations field-inputs)) (item))))
                           (py.. activations (size 0))))
                     -indices)]
    (py.. activations (index_fill_ 0 idxs-fresh 1))
    field-target))

(comment
  (torch/clamp_max_ (torch/tensor [0 2 1]) 1)
  (vacuum-babble (torch/zeros [3] :dtype torch/float) 0.5))

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

(defmulti interact (fn [op arg-map & _] op))

;; (defmethod interact :attracted
;;   [_ f & fields]
;;   (update f
;;           :weights
;;           (fn [w]
;;             (apply attracted w (map :weights fields)))))

(defmethod interact :production
  [_ field-target {:keys [production-factor]} & fields])

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




;;
;;
;;
;; inputs
;;

;; this is a kind of saliance
(defn resonate
  [weights inputs factor]
  ;; make the attractor basin that goes towards
  ;; inputs deeper
  ;;
  (torch/einsum "i,ij->ij"
                (torch/add (torch/mul inputs factor) 1)
                weights))

;; (defn resonate-update
;;   [field world-activations factor letter]
;;   (update field
;;           :weights
;;           resonate
;;           (torch/eq (pyutils/ensure-torch world-activations)
;;                     letter)
;;           factor))

;; ...
;; but the concept of excitability is much more simple

(defn normalize-excitability
  [excitability]
  (torch/where (torch/ne excitability 0)
               (torch/clamp excitability
                            excitability-threshold)
               excitability))

(comment
  (normalize-excitability
   (torch/tensor [0 0 1 0 0 0 0 0 1e-6] :dtype torch/float)))

(defn update-excitability
  [exc selection factor]
  (let [current-values (py/get-item exc selection)
        updated-values (torch/mul current-values factor)]
    (normalize-excitability
      (py/set-item! exc selection updated-values))))

(defn resonate-update
  [field world-activations factor letter]
  (update field
          :excitability
          update-excitability
          (torch/eq (pyutils/ensure-torch world-activations)
                    letter)
          factor))

;; -----------

(defn attenuation
  [excitablity activations factor]
  (if (zero? factor)
    excitablity
    (update-excitability excitablity
                         (torch/eq activations 1)
                         (/ 1 factor))))

;; --------

(comment
  (attenuation (torch/arange 3)
               (torch/tensor [false false true])
               2))

(defn attenuation-update
  "Attenuation with less than `attenuation-factor` 1 is an intrinsic excitability gain implementation.
  "
  [{:as s :keys [attenuation-factor activations]}]
  (update s
          :excitability
          attenuation
          activations
          attenuation-factor))

;; (defn directional-pull
;;   [excitablity direction pull-factor]
;;   ;; update the excitabilities with a gradient
;;   (let [size (py.. excitablity (size 0))
;;         size (long (Math/sqrt size))
;;         pull-factor 0.5]
;;     (normalize-excitability
;;       (py.. (torch/einsum
;;               "ij,i->ij"
;;               (py.. excitablity (view size size))
;;               (torch/add
;;                (torch/mul
;;                 (torch/add
;;                  (torch/arange size :device pyutils/*torch-device*)
;;                  1)
;;                 pull-factor)
;;                1))
;;         (view -1)))))

(defn directional-pull
  [excitablity direction pull-factor]
  ;; update the excitabilities with a gradient
  (let [size (py.. excitablity (size 0))
        size (long (Math/sqrt size))
        pull-factor 0.5]
    (normalize-excitability
     (py..
         (torch/einsum
          "ij,i->ij"
          (py.. excitablity (view size size))
          (torch/add
           (torch/mul
            (torch/add
             (torch/arange size :device pyutils/*torch-device*)
             1)
            pull-factor)
           1))
       (view -1)))))

(defn pull-update
  [direction {:as s :keys [excitability pull-factor]}]
  (update s
          :excitability
          directional-pull
          direction
          pull-factor))

(comment
  (torch/einsum "i,i->i" (torch/add (torch/mul inputs factor) 1)))

;; --------
(comment
  (def f (grid-field 3 [brownian-update]))
  (field-matrix 30)
  (field-matrix 3)

  (default-update
   (torch/mul
    (field-matrix 3)
    (py/set-item! (torch/ones [9 9]) 2 2))
   (torch/tensor [1 0 0 0 0 0 0 0 0] :dtype torch/float))

  (default-update
   (torch/tensor
    [[0 0 1]
     [0 0 1]
     [0 0 1]]
    :dtype
    torch/float)
   (torch/tensor [0 1 0] :dtype torch/float)))


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
  (default-update
   (brownian-motion (field-matrix 3))
   (torch/tensor [1 0 0 0 0 0 0 0 0] :dtype torch/float))


  (default-update
   (brownian-motion (field-matrix 3))
   (torch/tensor [1 0 0 0 0 0 0 0 0] :dtype torch/float))


  (def weights
    (resonate
     (torch/ones [9 9] :dtype torch/float)
     (torch/tensor [0 0 0 0 0 0 0 0 1] :dtype torch/bool)
     2))



  (def activations (torch/tensor [0 0 0
                                  0 0 1
                                  0 0 1] :dtype torch/float))

  (torch/mv weights activations)
  (torch/mv (torch/ones [9 9] :dtype torch/float) activations)

  (let [inputs (torch/mv weights activations)
        idxs (py.. (torch/topk inputs
                               (long (py.. (torch/sum
                                            activations)
                                       item)))
               -indices)
        _ (py.. activations (fill_ 0))
        _ (py.. activations (index_fill_ 0 idxs 1))]
    activations)


  (torch/einsum
   "i,ij->ij"
   (torch/add
    (torch/mul (torch/tensor [true false false]) 2)
    1)
   (torch/ones [3 3]))


  (torch/add
   (torch/mul (torch/tensor [true false false]) 2)
   1)

  (def w (torch/rand [3 3]))
  w
  ;; tensor([[0.4422, 0.9762, 0.4654],
  ;;       [0.0434, 0.3445, 0.5620],
  ;;       [0.2052, 0.1680, 0.4251]])

  (torch/einsum "i,ij->ij" (torch/tensor [1 1 2]) w)

  (torch/mv
   (resonate
    (torch/ones [9 9])
    (torch/tensor [false false false false false
                   false false false true])
    2)
   (torch/tensor [0 0 0 0 0 0 0 0 1] :dtype torch/float))
  ;; tensor([1., 1., 1., 1., 1., 1., 1., 1., 3.])




  )

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

  (f/sum (pyutils/torch->jvm (:activations field)))
  (f/sum
   (f/bit-and
    (f/<= 1 (pyutils/torch->jvm (:activations field)))
    (f/eq 1
          (dtt/->tensor
           (dtt/reshape
            (dtt/compute-tensor
             [30 30]
             (fn [i j]
               (if (and (< 10 i 20) (< 10 j 20)) 1.0 0.0))
             :float32)
            [(* 30 30)])))))

  ;; ---------



  (let [grid-width 3
        ->point (fn [i] [(fm/mod i grid-width)
                         (fm/quot i grid-width)])]
    (dtt/compute-tensor
     [(* grid-width grid-width) (* grid-width grid-width)]
     (fn [i j] [(l1-loss (->point i) (->point j)) :p1
                (->point i) :p2 (->point j) :i i :j j])
     :object))

  (require-python '[local_square_matrix :as lsm :reload true])


  (torch/allclose (lsm/local_square_matrix_torch 3)
                  (field-matrix 3))
  (torch/allclose (time (lsm/local_square_matrix_torch 30))
                  (time (field-matrix 10)))



  (fm/rem 10 2)

  (fm// 12 2)

  (torch/_foreach_max
   (torch/tensor [0 1 2])
   1)
  (torch/clamp_max)


  (torch/div (torch/tensor [0 1 2 3]) 2)

  (torch/divide (torch/tensor [0 1 2 3]) 2)

  (torch/fmod (torch/tensor [0 1 2 3]) 2)
  (torch/fmod (torch/tensor [0 1 2 3 4 5 6 8 9 10]) 5)


  (let
      [[x y]
       (torch/meshgrid (torch/arange 3) (torch/arange 3))
       positions
       (torch/stack [(py.. x (flatten)) (py.. y (flatten))]
                    :dim
                    1)
       diff
       (torch/subtract (torch/unsqueeze positions 1)
                       (torch/unsqueeze positions 0))
       distances
       (-> (torch/sum (torch/abs diff) :dim 2)
           ;; (torch/fmod (inc (fm// 3 2)))
           )]
    ;; positions
      (torch/unsqueeze positions 0)
      (torch/unsqueeze positions 1)
      diff
      distances)


  (let [grid-size 30
        [x y] (torch/meshgrid (torch/arange grid-size)
                              (torch/arange grid-size))
        positions (torch/stack [(py.. x (flatten))
                                (py.. y (flatten))]
                               :dim
                               1)
        diff (torch/subtract (torch/unsqueeze positions 1)
                             (torch/unsqueeze positions 0))
        toroidal_diff (torch/fmod (torch/add diff
                                             (torch/unsqueeze
                                              (torch/tensor
                                               [grid-size])
                                              0))
                                  grid-size)
        toroidal_diff_min
        (torch/min toroidal_diff
                   (torch/subtract grid-size toroidal_diff))
        distances
        (torch/sum (torch/abs toroidal_diff_min) :dim 2)
        adjacency (torch/le distances 2)
        adjacency (py.. adjacency (to :dtype torch/float))]
    adjacency))
