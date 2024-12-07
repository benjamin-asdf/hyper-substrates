(ns arc.ca
  (:require
    [bennischwerdtner.pyutils :as pyutils :refer
     [*torch-device*]]
    [tech.v3.datatype.functional :as f]
    [tech.v3.tensor :as dtt]
    [fastmath.random :as fm.rand]
    [fastmath.core :as fm]
    [bennischwerdtner.hd.codebook-item-memory :as codebook]
    [bennischwerdtner.hd.data :as hdd]
    [bennischwerdtner.hd.core :as hd]
    [libpython-clj2.require :refer [require-python]]
    [libpython-clj2.python :refer [py. py..] :as py]))

(require-python 'torch)
(require-python '[torch.nn.functional :as F])

(alter-var-root
 #'hdd/*item-memory*
 (constantly (codebook/codebook-item-memory 1000)))

;; --------------------------------------------
(def grid
  (torch/tensor
   [[5 5 5 5 0 5 5 5 5 0 5 5 5 5]
    [5 5 5 5 0 0 5 5 0 0 5 0 0 5]
    [5 0 0 5 0 0 5 5 0 0 5 0 0 5]
    [5 0 0 5 0 5 5 5 5 0 5 5 5 5]]))



(comment
  (torch/ge (torch/rand [10]) 0)



  (/ 56 (* 20 20))

  (/ 56 (* 20 20))

  (let [output-shape [20 20]
        output-size (apply * output-shape)
        grid (torch/tensor
              [[5 5 5 5 0 5 5 5 5 0 5 5 5 5]
               [5 5 5 5 0 0 5 5 0 0 5 0 0 5]
               [5 0 0 5 0 0 5 5 0 0 5 0 0 5]
               [5 0 0 5 0 5 5 5 5 0 5 5 5 5]])

        input-size (py.. grid numel)
        rule
        ;; (py.. (torch/ge (torch/rand [(apply * output-shape) input-size])
        ;;                 0.9)
        ;;   (to :dtype torch/float))
        (py..
            (torch/ge
             (torch/rand [output-size input-size])
             (- 1 (/ input-size output-size 5)))
          (to :dtype torch/float))]
    (torch/mv
     rule
     (py..
         (torch/eq grid 5)
         (view -1)
         (to :dtype torch/float))))








  )

;; tensor([0, 1, 0, 0, 1, 0, 0, 1, 0])
;; i->j
;; 9x9
;; ---------------------------------------

(defn rand-update-rule
  []
  (py.. (torch/ge (torch/rand [9 9]) 0.9)
    (to :dtype torch/float)))


(defn rand-update-rule-2
  [input-size output-size]
  (py.. (torch/ge
          (torch/rand [output-size input-size])
          (- 1 (/ input-size (* output-size output-size))))
        (to :dtype torch/float)))

(defn gene->input-color
  [gene]
  (first (hdd/cleanup* (hdd/clj->vsa* [:. gene
                                       :input-color]))))

(defn gene->output-color [gene]
  (first
   (hdd/cleanup*
    (hdd/clj->vsa*
     [:. gene :output-color]))))

(comment
  (gene->output-color
   (hdd/clj->vsa*
    [:+
     [:* 0 :output-color]])))

(defn ->gene
  []
  (hdd/clj->vsa*
   [:+
    [:* (rand-int 10) :output-color]
    [:* (rand-int 10) :input-color]]))

(defn gene->update-rule [gene] (rand-update-rule))

(defn update-function
  [grid]
  (let [gene (->gene)

        output-shape [(inc (rand-int 30))
                      (inc (rand-int 30))]

        output-shape (into []
                           (py.. grid (size)))


        update-rule (rand-update-rule-2 (py.. grid numel)
                                        (apply *
                                               output-shape))
        output-color (if false
                       ;; (zero? (fm.rand/flip 0.5))
                       -1
                       (or (gene->output-color gene) 0))
        input-color (or (gene->input-color gene) 0)]
    (-> (torch/mv update-rule
                  (py.. (torch/eq grid input-color)
                        (view -1)
                        (to :dtype torch/float)))
        (torch/ge 1)
        (py.. (mul output-color)
          (reshape (py/->py-tuple output-shape))))))

(comment
  (let [output-shape [(inc (rand-int 30))
                      (inc (rand-int 30))]
        ;; input-shape
        ;; (into [] (py.. grid (shape)))
        input-size (py.. grid numel)]
    (py.. (torch/ge (torch/rand [(apply * output-shape) input-size])
                    0.9)
      (to :dtype torch/float))))
