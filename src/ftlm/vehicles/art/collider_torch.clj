(ns ftlm.vehicles.art.collider-torch
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

(do (require-python '[numpy :as np])
    (require-python '[torch :as torch])
    (require-python '[torch.sparse :as torch.sparse])
    (require-python '[builtins :as builtins])
    (require '[libpython-clj2.python.np-array]))

(defn pairwise-distance
  "Returns a matrix with shape

  [n n],

  These are the (cartesian or how that is called) distances for each element in `positions`.
  Naturally, for all i,j: d = distances[i,j], if (i == j) then d == 0.
  "
  [positions]
  (-> (torch/subtract (torch/unsqueeze positions 1)
                      (torch/unsqueeze positions 0))
      (torch/pow 2)
      (torch/sum :dim -1)
      (torch/sqrt)))

(defn detect-collisions
  [positions radii]
  (let [out (torch/zeros [(py.. positions (size 0)) 2]
                         :device *torch-device*
                         :dtype torch/long)]
    (py/with-gil-stack-rc-context
      (let [distances (pairwise-distance positions)
            sum-radii (-> (torch/add
                            (torch/unsqueeze radii 1)
                            (torch/unsqueeze radii 0))
                          ;;  Use upper triangular to
                          ;;  avoid self-collisions and
                          ;;  duplicates
                          (torch/triu 1))]
        (-> (torch/lt distances sum-radii)
            (torch/nonzero :out out))))
    out))

;;
;; returns a list of pairs [idx-a idx-b]
;;

(defn collisions
  [positions radii]
  (let [positions (torch/tensor (vec positions)
                                :device
                                  pyutils/*torch-device*
                                :dtype torch/float)
        radii (torch/tensor (vec radii)
                            :device pyutils/*torch-device*
                            :dtype torch/float)]
    (pyutils/torch->jvm (detect-collisions positions
                                           radii))))
